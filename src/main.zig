const std = @import("std");

const compile = @import("compile.zig");
const Glslc = compile.Glslc;

const Shadertoy = @import("shadertoy.zig");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;

const mach = @import("mach");

// using wgpu instead of mach.gpu for better lsp results
const gpu = mach.wgpu;

// The global list of Mach modules registered for use in our application.
pub const modules = .{
    mach.Core,
    App,
    Renderer,
    Gui,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

pub const config = .{
    .paths = .{
        .toys = "./toys",
        .shadertoys = "./toys/shadertoy",
        .zhadertoys = "./toys/zhadertoy",
        .playground = "./shaders",
    },
};

const Renderer = struct {
    pub const name = .renderer;
    pub const systems = .{
        .init = .{ .handler = init },
        .deinit = .{ .handler = deinit },
        .tick = .{ .handler = tick },
        .render = .{ .handler = render },
        .update_shaders = .{ .handler = update_shaders },
    };
    pub const Mod = mach.Mod(@This());

    const StateUniform = extern struct {
        render_width: u32,
        render_height: u32,
        display_width: u32,
        display_height: u32,
        windowless: u32 = 0,
        time: f32 = 0.0,
        cursor_x: f32 = 0.0,
        cursor_y: f32 = 0.0,
        scroll: f32 = 0.0,

        mouse_left: u32 = 0,
        mouse_right: u32 = 0,
        mouse_middle: u32 = 0,
    };
    // uniform bind group offset must be 256-byte aligned
    const uniform_offset = 256;

    const IUniform = struct {
        ptr: *anyopaque,
        vtable: *const struct {
            update: *const fn (any_self: *anyopaque, encoder: *gpu.CommandEncoder) void,
            deinit: *const fn (any_self: *anyopaque, alloc: std.mem.Allocator) void,
            bindGroupLayoutEntry: *const fn (any_self: *anyopaque, binding: u32) gpu.BindGroupLayout.Entry,
            bindGroupEntry: *const fn (any_self: *anyopaque, binding: u32) gpu.BindGroup.Entry,
        },

        fn bindGroupLayoutEntry(self: *@This(), binding: u32) gpu.BindGroupLayout.Entry {
            return self.vtable.bindGroupLayoutEntry(self.ptr, binding);
        }
        fn bindGroupEntry(self: *@This(), binding: u32) gpu.BindGroup.Entry {
            return self.vtable.bindGroupEntry(self.ptr, binding);
        }
        fn update(self: *@This(), encoder: *gpu.CommandEncoder) void {
            self.vtable.update(self.ptr, encoder);
        }
        fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            self.vtable.deinit(self.ptr, alloc);
        }
    };
    fn Buffer(typ: anytype) type {
        return struct {
            val: typ,
            buffer: *gpu.Buffer,
            label: [*:0]const u8,

            fn new(comptime nam: []const u8, val: typ, alloc: std.mem.Allocator, device: *gpu.Device) !*@This() {
                const label = nam ++ ".init";
                const buffer = device.createBuffer(&.{
                    .label = label ++ " uniform buffer",
                    .usage = .{ .copy_dst = true, .uniform = true },
                    .size = @sizeOf(typ) * uniform_offset, // HUH: why is this size*uniform_offset? it should be the next multiple of uniform_offset
                    .mapped_at_creation = .false,
                });
                const buff = try alloc.create(@This());
                buff.* = .{
                    .val = val,
                    .buffer = buffer,
                    .label = label,
                };
                return buff;
            }
            fn bindGroupLayoutEntry(_: *@This(), binding: u32) gpu.BindGroupLayout.Entry {
                return gpu.BindGroupLayout.Entry.buffer(
                    binding,
                    .{
                        .vertex = true,
                        .fragment = true,
                        .compute = true,
                    },
                    .uniform,
                    true,
                    0,
                );
            }
            fn bindGroupEntry(self: *@This(), binding: u32) gpu.BindGroup.Entry {
                return gpu.BindGroup.Entry.buffer(binding, self.buffer, 0, @sizeOf(typ));
            }
            fn update(self: *@This(), encoder: *gpu.CommandEncoder) void {
                encoder.writeBuffer(self.buffer, 0, &[_]typ{self.val});
            }
            fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
                self.buffer.release();
                alloc.destroy(self);
            }
            fn interface(self: *@This()) IUniform {
                return .{
                    .ptr = self,
                    .vtable = &.{
                        .update = @ptrCast(&update),
                        .deinit = @ptrCast(&@This().deinit),
                        .bindGroupLayoutEntry = @ptrCast(&bindGroupLayoutEntry),
                        .bindGroupEntry = @ptrCast(&bindGroupEntry),
                    },
                };
            }
        };
    }
    const Binding = struct {
        label: [*:0]const u8,
        group: ?*gpu.BindGroup = null,
        layout: ?*gpu.BindGroupLayout = null,
        buffers: std.ArrayListUnmanaged(IUniform) = .{},

        fn add(self: *@This(), buffer: anytype, alloc: std.mem.Allocator) !void {
            const itf: IUniform = buffer.interface();
            try self.buffers.append(alloc, itf);
        }
        fn init(self: *@This(), label: [*:0]const u8, device: *gpu.Device, alloc: std.mem.Allocator) !void {
            var layout_entries = std.ArrayListUnmanaged(gpu.BindGroupLayout.Entry){};
            defer layout_entries.deinit(alloc);
            var bind_group_entries = std.ArrayListUnmanaged(gpu.BindGroup.Entry){};
            defer bind_group_entries.deinit(alloc);

            for (self.buffers.items, 0..) |*buf, i| {
                try layout_entries.append(alloc, buf.bindGroupLayoutEntry(@intCast(i)));

                try bind_group_entries.append(alloc, buf.bindGroupEntry(@intCast(i)));
            }

            const bind_group_layout = device.createBindGroupLayout(
                &gpu.BindGroupLayout.Descriptor.init(.{
                    .label = label,
                    .entries = layout_entries.items,
                }),
            );

            const bind_group = device.createBindGroup(
                &gpu.BindGroup.Descriptor.init(.{
                    .label = label,
                    .layout = bind_group_layout,
                    .entries = bind_group_entries.items,
                }),
            );
            self.group = bind_group;
            self.layout = bind_group_layout;
        }
        fn update(self: *@This(), encoder: *gpu.CommandEncoder) void {
            for (self.buffers.items) |*buf| {
                buf.update(encoder);
            }
        }
        fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            if (self.layout) |layout| layout.release();
            if (self.group) |group| group.release();

            for (self.buffers.items) |*buf| {
                buf.deinit(alloc);
            }
            self.buffers.deinit(alloc);
        }
    };

    const Vec4 = extern struct {
        x: f32 = 0.0,
        y: f32 = 0.0,
        z: f32 = 0.0,
        w: f32 = 0.0,
    };
    const Vec3 = extern struct {
        x: f32 = 0.0,
        y: f32 = 0.0,
        z: f32 = 0.0,
    };
    const ShadertoyUniforms = extern struct {
        time: f32 = 0.0,
        time_delta: f32 = 0.0,
        frame: u32 = 0,
        width: u32,
        height: u32,
        mouse_x: f32 = 0.0,
        mouse_y: f32 = 0.0,

        padding: u64 = 0,
    };
    const ShadertoyUniformBuffers = struct {
        uniforms: *Buffer(ShadertoyUniforms),

        fn init(core_mod: *mach.Core.Mod) !@This() {
            const core: *mach.Core = core_mod.state();
            const device = core.device;
            const height = core_mod.get(core.main_window, .height).?;
            const width = core_mod.get(core.main_window, .width).?;
            return .{
                .uniforms = try Buffer(ShadertoyUniforms).new(@tagName(name) ++ " uniforms", .{
                    .width = width,
                    .height = height,
                }, allocator, device),
            };
        }
    };

    // on shader change
    // - compile frag shaders 5 times (1 image, 4 buffers)
    //   - each with different definitions which include different files
    // - thus needs 5 diff pipelines
    // - needs 5 * 2 bind groups
    //   - 2 bind groups for each shader. 2 cuz we need to swap textures each frame
    //   - all textures * 2
    const PlaygroundRenderInfo = struct {
        const ImagePass = struct {
            pipeline: *gpu.RenderPipeline,
            bind_group: struct {
                current: *gpu.BindGroup,
                last_frame: *gpu.BindGroup,
            },
            bind_group_layout: *gpu.BindGroupLayout,
            inputs: Pass.Inputs,

            fn init(
                at: *Shadertoy.ActiveToy,
                render_config: *const Config,
                core_mod: *mach.Core.Mod,
                channels: *Channels,
                binding: *Binding,
                empty_input: *EmptyInput,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var inputs = Pass.Inputs.init(device, &at.passes.image.inputs);
                errdefer inputs.release();
                var bind = try Pass.create_bind_group(device, binding, channels, &inputs, &at.passes.image.inputs, empty_input);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try Pass.create_pipeline(device, render_config, &inputs, at.has_common, bind.layout, core_mod.get(core.main_window, .framebuffer_format).?, .image);
                errdefer pipeline.release();

                return .{
                    .inputs = inputs,
                    .bind_group_layout = bind.layout,
                    .bind_group = .{
                        .current = bind.group1,
                        .last_frame = bind.group2,
                    },
                    .pipeline = pipeline,
                };
            }

            fn render(self: *@This(), encoder: *gpu.CommandEncoder, screen: *gpu.TextureView) void {
                Pass._render_pass(self.pipeline, self.bind_group.current, encoder, screen);
            }

            fn release(self: *@This()) void {
                self.bind_group.current.release();
                self.bind_group.last_frame.release();
                self.pipeline.release();
            }
        };
        const Pass = struct {
            const Input = struct {
                typ: Shadertoy.ActiveToy.Buf,
                sampler: *gpu.Sampler,

                fn init(device: *gpu.Device, input: ?Shadertoy.ActiveToy.Input) ?@This() {
                    // TODO: use input.?.sampler
                    if (input) |inp| {
                        return .{
                            .typ = inp.typ,
                            .sampler = device.createSampler(&.{}),
                        };
                    } else {
                        return null;
                    }
                }

                fn release(self: *@This()) void {
                    self.sampler.release();
                }
            };
            const Inputs = struct {
                input1: ?Input,
                input2: ?Input,
                input3: ?Input,
                input4: ?Input,

                fn init(device: *gpu.Device, inputs: *Shadertoy.ActiveToy.Inputs) @This() {
                    return .{
                        .input1 = Input.init(device, inputs.input1),
                        .input2 = Input.init(device, inputs.input2),
                        .input3 = Input.init(device, inputs.input3),
                        .input4 = Input.init(device, inputs.input4),
                    };
                }

                fn release(self: *@This()) void {
                    if (self.input1) |*inp| inp.release();
                    if (self.input2) |*inp| inp.release();
                    if (self.input3) |*inp| inp.release();
                    if (self.input4) |*inp| inp.release();
                }
            };

            pipeline: *gpu.RenderPipeline,
            bind_group: struct {
                current: *gpu.BindGroup,
                last_frame: *gpu.BindGroup,
            },
            bind_group_layout: *gpu.BindGroupLayout,
            output: Shadertoy.ActiveToy.Buf,
            inputs: Inputs,

            fn init(
                at: *Shadertoy.ActiveToy,
                render_config: *const Config,
                core_mod: *mach.Core.Mod,
                pass: *Shadertoy.ActiveToy.BufferPass,
                channels: *Channels,
                binding: *Binding,
                include: enum { buffer1, buffer2, buffer3, buffer4 },
                empty_input: *EmptyInput,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var inputs = Inputs.init(device, &pass.inputs);
                var bind = try create_bind_group(device, binding, channels, &inputs, &pass.inputs, empty_input);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try create_pipeline(device, render_config, &inputs, at.has_common, bind.layout, .rgba32_float, switch (include) {
                    .buffer1 => .buffer1,
                    .buffer2 => .buffer2,
                    .buffer3 => .buffer3,
                    .buffer4 => .buffer4,
                });
                errdefer pipeline.release();

                return .{
                    .inputs = inputs,
                    .output = pass.output.typ,
                    .bind_group_layout = bind.layout,
                    .bind_group = .{
                        .current = bind.group1,
                        .last_frame = bind.group2,
                    },
                    .pipeline = pipeline,
                };
            }

            fn create_bind_group(
                device: *gpu.Device,
                binding: *Binding,
                _channels: *Channels,
                _inputs: *Inputs,
                _at_inputs: *Shadertoy.ActiveToy.Inputs,
                empty_input: *EmptyInput,
            ) !struct { group1: *gpu.BindGroup, group2: *gpu.BindGroup, layout: *gpu.BindGroupLayout } {
                const alloc = allocator;
                var layout_entries = std.ArrayListUnmanaged(gpu.BindGroupLayout.Entry){};
                defer layout_entries.deinit(alloc);
                var bind_group_entries = std.ArrayListUnmanaged(gpu.BindGroup.Entry){};
                defer bind_group_entries.deinit(alloc);

                for (binding.buffers.items, 0..) |*buf, i| {
                    try layout_entries.append(alloc, buf.bindGroupLayoutEntry(@intCast(i)));

                    try bind_group_entries.append(alloc, buf.bindGroupEntry(@intCast(i)));
                }

                const Fn = struct {
                    layouts: @TypeOf(layout_entries),
                    groups: @TypeOf(bind_group_entries),
                    empty: @TypeOf(empty_input),
                    alloc: @TypeOf(alloc),

                    const visibility = .{
                        .vertex = true,
                        .fragment = true,
                        .compute = true,
                    };

                    fn add(self: *@This(), view: *gpu.TextureView, sampler: *gpu.Sampler) !void {
                        try self.layouts.append(self.alloc, gpu.BindGroupLayout.Entry.texture(
                            @intCast(self.layouts.items.len),
                            visibility,
                            .unfilterable_float,
                            .dimension_2d,
                            false,
                        ));
                        try self.groups.append(self.alloc, gpu.BindGroup.Entry.textureView(@intCast(self.groups.items.len), view));
                        try self.layouts.append(self.alloc, gpu.BindGroupLayout.Entry.sampler(
                            @intCast(self.layouts.items.len),
                            visibility,
                            .non_filtering,
                        ));
                        try self.groups.append(self.alloc, gpu.BindGroup.Entry.sampler(@intCast(self.groups.items.len), sampler));
                    }

                    fn add_all(self: *@This(), channels: *Channels, inputs: *Inputs, at_inputs: *Shadertoy.ActiveToy.Inputs, swap_tex: bool) !void {
                        if (inputs.input1) |inp| {
                            const tex = channels.get(inp.typ);
                            var curr: *gpu.TextureView = tex.current.view;
                            var last: *gpu.TextureView = tex.last_frame.view;
                            if (swap_tex) {
                                std.mem.swap(*gpu.TextureView, &curr, &last);
                            }
                            const view = if (at_inputs.input1.?.from_current_frame) curr else last;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input2) |inp| {
                            const tex = channels.get(inp.typ);
                            var curr: *gpu.TextureView = tex.current.view;
                            var last: *gpu.TextureView = tex.last_frame.view;
                            if (swap_tex) {
                                std.mem.swap(*gpu.TextureView, &curr, &last);
                            }
                            const view = if (at_inputs.input2.?.from_current_frame) curr else last;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input3) |inp| {
                            const tex = channels.get(inp.typ);
                            var curr: *gpu.TextureView = tex.current.view;
                            var last: *gpu.TextureView = tex.last_frame.view;
                            if (swap_tex) {
                                std.mem.swap(*gpu.TextureView, &curr, &last);
                            }
                            const view = if (at_inputs.input3.?.from_current_frame) curr else last;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input4) |inp| {
                            const tex = channels.get(inp.typ);
                            var curr: *gpu.TextureView = tex.current.view;
                            var last: *gpu.TextureView = tex.last_frame.view;
                            if (swap_tex) {
                                std.mem.swap(*gpu.TextureView, &curr, &last);
                            }
                            const view = if (at_inputs.input4.?.from_current_frame) curr else last;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                    }
                };

                var fnn1 = Fn{
                    .layouts = try layout_entries.clone(allocator),
                    .groups = try bind_group_entries.clone(allocator),
                    .empty = empty_input,
                    .alloc = alloc,
                };
                defer {
                    fnn1.layouts.deinit(alloc);
                    fnn1.groups.deinit(alloc);
                }
                try fnn1.add_all(_channels, _inputs, _at_inputs, false);

                const bind_group_layout = device.createBindGroupLayout(
                    &gpu.BindGroupLayout.Descriptor.init(.{
                        .label = "bind grpu",
                        .entries = fnn1.layouts.items,
                    }),
                );
                const bind_group1 = device.createBindGroup(
                    &gpu.BindGroup.Descriptor.init(.{
                        .label = "bind grpup adfasf",
                        .layout = bind_group_layout,
                        .entries = fnn1.groups.items,
                    }),
                );

                var fnn2 = Fn{
                    .layouts = try layout_entries.clone(alloc),
                    .groups = try bind_group_entries.clone(alloc),
                    .empty = empty_input,
                    .alloc = alloc,
                };
                defer {
                    fnn2.layouts.deinit(alloc);
                    fnn2.groups.deinit(alloc);
                }
                try fnn2.add_all(_channels, _inputs, _at_inputs, true);

                const bind_group2 = device.createBindGroup(
                    &gpu.BindGroup.Descriptor.init(.{
                        .label = "bind grpup adff",
                        .layout = bind_group_layout,
                        .entries = fnn2.groups.items,
                    }),
                );

                return .{
                    .group1 = bind_group1,
                    .group2 = bind_group2,
                    .layout = bind_group_layout,
                };
            }

            fn create_pipeline(
                device: *gpu.Device,
                render_config: *const Config,
                inputs: *Inputs,
                has_common: bool,
                bg_layout: *gpu.BindGroupLayout,
                format: gpu.Texture.Format,
                include: enum { image, buffer1, buffer2, buffer3, buffer4 },
            ) !*gpu.RenderPipeline {
                var definitions = std.ArrayList([]const u8).init(allocator);
                defer {
                    for (definitions.items) |def| {
                        allocator.free(def);
                    }
                    definitions.deinit();
                }
                if (has_common) {
                    try definitions.append(try allocator.dupe(u8, "ZHADER_COMMON"));
                }

                // NOTE: webgpu does not have a simple way to flip texture without copying it.
                // shadertoy's buffer inputs always have vflip = true
                // on shadertoy, texture fetches have y coord flipped compared to webgpu
                // this hack just flips the y coord on buffer's mainImage fragCoord param. might fail if shadertoy
                //  buffer vflip = false
                switch (include) {
                    .image => {},
                    .buffer1, .buffer2, .buffer3, .buffer4 => {
                        try definitions.append(try allocator.dupe(u8, "ZHADER_VFLIP"));
                    },
                }

                _ = inputs;
                // if (inputs.input1) |_| try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL0"));
                // if (inputs.input2) |_| try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL1"));
                // if (inputs.input3) |_| try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL2"));
                // if (inputs.input4) |_| try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL3"));

                try definitions.append(try std.fmt.allocPrint(allocator, "ZHADER_INCLUDE_{s}", .{switch (include) {
                    .image => "IMAGE",
                    .buffer1 => "BUF1",
                    .buffer2 => "BUF2",
                    .buffer3 => "BUF3",
                    .buffer4 => "BUF4",
                }}));

                var compiler = Glslc.Compiler{ .opt = render_config.shader_compile_opt };
                compiler.stage = .vertex;
                const vert: Glslc.Compiler.Code = .{ .path = .{
                    .main = "./shaders/vert.glsl",
                    .include = &[_][]const u8{config.paths.playground},
                    .definitions = definitions.items,
                } };
                // if (render_config.shader_dump_assembly) {
                //     _ = compiler.dump_assembly(allocator, &vert);
                // }
                const vertex_bytes = try compiler.compile(
                    allocator,
                    &vert,
                    .spirv,
                );
                defer allocator.free(vertex_bytes);
                const vertex_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
                    .next_in_chain = .{ .spirv_descriptor = &.{
                        .code_size = @intCast(vertex_bytes.len),
                        .code = vertex_bytes.ptr,
                    } },
                    .label = "vert.glsl",
                });
                defer vertex_shader_module.release();

                compiler.stage = .fragment;
                const frag: Glslc.Compiler.Code = .{ .path = .{
                    .main = "./shaders/frag.glsl",
                    .include = &[_][]const u8{config.paths.playground},
                    .definitions = definitions.items,
                } };
                if (render_config.shader_dump_assembly) {
                    compiler.dump_assembly(allocator, &frag) catch {};
                }
                const frag_bytes = try compiler.compile(
                    allocator,
                    &frag,
                    .spirv,
                );
                defer allocator.free(frag_bytes);
                const frag_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
                    .next_in_chain = .{ .spirv_descriptor = &.{
                        .code_size = @intCast(frag_bytes.len),
                        .code = frag_bytes.ptr,
                    } },
                    .label = "frag.glsl",
                });
                defer frag_shader_module.release();

                const color_target = gpu.ColorTargetState{
                    .format = format,
                    // .blend = &gpu.BlendState{},
                    .write_mask = gpu.ColorWriteMaskFlags.all,
                };
                const fragment = gpu.FragmentState.init(.{
                    .module = frag_shader_module,
                    .entry_point = "main",
                    .targets = &.{color_target},
                });
                const vertex = gpu.VertexState{
                    .module = vertex_shader_module,
                    .entry_point = "main",
                };

                const bind_group_layouts = [_]*gpu.BindGroupLayout{bg_layout};
                const pipeline_layout = device.createPipelineLayout(&gpu.PipelineLayout.Descriptor.init(.{
                    .label = "buffer render pipeline",
                    .bind_group_layouts = &bind_group_layouts,
                }));
                defer pipeline_layout.release();
                const pipeline = device.createRenderPipeline(&gpu.RenderPipeline.Descriptor{
                    .label = "buffer render pipeline",
                    .fragment = &fragment,
                    .layout = pipeline_layout,
                    .vertex = vertex,
                });

                return pipeline;
            }

            fn swap(self: *@This()) void {
                std.mem.swap(*gpu.BindGroup, &self.bind_group.current, &self.bind_group.last_frame);
            }
            fn render(self: *@This(), encoder: *gpu.CommandEncoder, target: *gpu.TextureView) void {
                _render_pass(self.pipeline, self.bind_group.current, encoder, target);
            }

            fn _render_pass(
                pipeline: *gpu.RenderPipeline,
                bg: *gpu.BindGroup,
                encoder: *gpu.CommandEncoder,
                target: *gpu.TextureView,
            ) void {
                const color_attachments = [_]gpu.RenderPassColorAttachment{.{
                    .view = target,
                    .clear_value = gpu.Color{
                        .r = 40.0 / 255.0,
                        .g = 40.0 / 255.0,
                        .b = 40.0 / 255.0,
                        .a = 1,
                    },
                    .load_op = .clear,
                    .store_op = .store,
                }};
                const render_pass = encoder.beginRenderPass(&gpu.RenderPassDescriptor.init(.{
                    .label = "playground buffer render pass",
                    .color_attachments = &color_attachments,
                }));
                defer render_pass.release();

                render_pass.setPipeline(pipeline);
                render_pass.setBindGroup(0, bg, &.{0});
                render_pass.draw(6, 1, 0, 0);
                render_pass.end();
            }

            fn release(self: *@This()) void {
                self.bind_group.last_frame.release();
                self.bind_group.current.release();
                self.pipeline.release();
            }
        };
        const Channel = struct {
            const Tex = struct {
                texture: *gpu.Texture,
                view: *gpu.TextureView,

                fn release(self: *@This()) void {
                    self.view.release();
                    self.texture.destroy();
                }
            };
            current: Tex,
            last_frame: Tex,

            fn init(device: *gpu.Device, size: gpu.Extent3D) @This() {
                const tex_desc = gpu.Texture.Descriptor.init(.{
                    .size = size,
                    .dimension = .dimension_2d,
                    .usage = .{
                        .texture_binding = true,
                        .render_attachment = true,
                    },
                    .format = .rgba32_float,
                    // .view_format = &[_]_{},
                });
                const curr = device.createTexture(&tex_desc);
                const last = device.createTexture(&tex_desc);

                return .{
                    .current = .{
                        .texture = curr,
                        .view = curr.createView(&.{}),
                    },
                    .last_frame = .{
                        .texture = last,
                        .view = last.createView(&.{}),
                    },
                };
            }

            fn swap(self: *@This()) void {
                std.mem.swap(Tex, &self.current, &self.last_frame);
            }

            fn release(self: *@This()) void {
                self.current.release();
                self.last_frame.release();
            }
        };
        const Channels = struct {
            bufferA: Channel,
            bufferB: Channel,
            bufferC: Channel,
            bufferD: Channel,

            fn init(device: *gpu.Device, size: gpu.Extent3D) @This() {
                return .{
                    .bufferA = Channel.init(device, size),
                    .bufferB = Channel.init(device, size),
                    .bufferC = Channel.init(device, size),
                    .bufferD = Channel.init(device, size),
                };
            }

            fn release(self: *@This()) void {
                self.bufferA.release();
                self.bufferB.release();
                self.bufferC.release();
                self.bufferD.release();
            }

            fn get(self: *@This(), buf: Shadertoy.ActiveToy.Buf) *Channel {
                switch (buf) {
                    .BufferA => return &self.bufferA,
                    .BufferB => return &self.bufferB,
                    .BufferC => return &self.bufferC,
                    .BufferD => return &self.bufferD,
                }
            }
            fn view(self: *@This(), buf: Shadertoy.ActiveToy.Buf) *gpu.TextureView {
                return self.get(buf).current.view;
            }
        };

        const EmptyInput = struct {
            tex: Channel.Tex,
            sampler: *gpu.Sampler,

            fn release(self: *@This()) void {
                self.sampler.release();
                self.tex.release();
            }
        };
        pass: struct {
            image: ImagePass,
            buffer1: ?Pass,
            buffer2: ?Pass,
            buffer3: ?Pass,
            buffer4: ?Pass,

            empty_input: EmptyInput,

            fn init() !@This() {
                return .{};
            }
        },
        channels: Channels,
        binding: Binding,
        uniforms: ShadertoyUniformBuffers,

        fn init(at: *Shadertoy.ActiveToy, render_config: *const Config, core_mod: *mach.Core.Mod) !@This() {
            const core: *mach.Core = core_mod.state();
            const device: *gpu.Device = core.device;

            const uniforms = try ShadertoyUniformBuffers.init(core_mod);

            var binding = Binding{ .label = "fajdl;f;jda" };
            errdefer binding.deinit(allocator);
            // LEAK: if .add() errors, those uniforms don't get deinited
            try binding.add(uniforms.uniforms, allocator);
            // TODO:
            // try binding.init(@tagName(name), device, allocator);

            const height = core_mod.get(core.main_window, .height).?;
            const width = core_mod.get(core.main_window, .width).?;
            var channels = Channels.init(device, .{
                .width = width,
                .height = height,
                .depth_or_array_layers = 1,
            });
            errdefer channels.release();

            const empty_tex = device.createTexture(&gpu.Texture.Descriptor.init(.{
                .size = .{
                    .width = 1,
                    .height = 1,
                    .depth_or_array_layers = 1,
                },
                .dimension = .dimension_2d,
                .usage = .{
                    .texture_binding = true,
                },
                .format = .rgba32_float,
            }));
            var empty_input = EmptyInput{
                .tex = .{
                    .texture = empty_tex,
                    .view = empty_tex.createView(&.{}),
                },
                .sampler = device.createSampler(&.{}),
            };
            errdefer empty_input.release();
            var image_pass = try ImagePass.init(at, render_config, core_mod, &channels, &binding, &empty_input);
            errdefer image_pass.release();
            var pass1 = if (at.passes.buffer1) |*pass| try Pass.init(at, render_config, core_mod, pass, &channels, &binding, .buffer1, &empty_input) else null;
            errdefer if (pass1) |*pass| pass.release();
            var pass2 = if (at.passes.buffer2) |*pass| try Pass.init(at, render_config, core_mod, pass, &channels, &binding, .buffer2, &empty_input) else null;
            errdefer if (pass2) |*pass| pass.release();
            var pass3 = if (at.passes.buffer3) |*pass| try Pass.init(at, render_config, core_mod, pass, &channels, &binding, .buffer3, &empty_input) else null;
            errdefer if (pass3) |*pass| pass.release();
            var pass4 = if (at.passes.buffer4) |*pass| try Pass.init(at, render_config, core_mod, pass, &channels, &binding, .buffer4, &empty_input) else null;
            errdefer if (pass4) |*pass| pass.release();

            return .{
                .pass = .{
                    .image = image_pass,
                    .buffer1 = pass1,
                    .buffer2 = pass2,
                    .buffer3 = pass3,
                    .buffer4 = pass4,
                    .empty_input = empty_input,
                },
                .uniforms = uniforms,
                .binding = binding,
                .channels = channels,
            };
        }

        fn update(self: *@This(), at: *Shadertoy.ActiveToy, render_config: *const Config, core_mod: *mach.Core.Mod, ev: Shadertoy.ToyMan.UpdateEvent) !void {
            switch (ev) {
                .Buffer1 => {
                    var buf1 = at.passes.buffer1 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer1.?;
                    const new_pass = try Pass.init(at, render_config, core_mod, &buf1, &self.channels, &self.binding, .buffer1, &self.pass.empty_input);
                    self.pass.buffer1 = new_pass;
                    pass.release();
                },
                .Buffer2 => {
                    var buf2 = at.passes.buffer2 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer2.?;
                    const new_pass = try Pass.init(at, render_config, core_mod, &buf2, &self.channels, &self.binding, .buffer2, &self.pass.empty_input);
                    self.pass.buffer2 = new_pass;
                    pass.release();
                },
                .Buffer3 => {
                    var buf3 = at.passes.buffer3 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer3.?;
                    const new_pass = try Pass.init(at, render_config, core_mod, &buf3, &self.channels, &self.binding, .buffer3, &self.pass.empty_input);
                    self.pass.buffer3 = new_pass;
                    pass.release();
                },
                .Buffer4 => {
                    var buf4 = at.passes.buffer4 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer4.?;
                    const new_pass = try Pass.init(at, render_config, core_mod, &buf4, &self.channels, &self.binding, .buffer4, &self.pass.empty_input);
                    self.pass.buffer4 = new_pass;
                    pass.release();
                },
                .Image => {
                    var pass = self.pass.image;
                    const new_pass = try ImagePass.init(at, render_config, core_mod, &self.channels, &self.binding, &self.pass.empty_input);
                    self.pass.image = new_pass;
                    pass.release();
                },
                .All => {
                    var old_self = self.*;
                    const new_self = try @This().init(at, render_config, core_mod);
                    self.* = new_self;
                    old_self.release();
                },
            }
        }

        fn swap(self: *@This()) void {
            self.channels.bufferA.swap();
            self.channels.bufferB.swap();
            self.channels.bufferC.swap();
            self.channels.bufferD.swap();
            if (self.pass.buffer1) |*buf| buf.swap();
            if (self.pass.buffer2) |*buf| buf.swap();
            if (self.pass.buffer3) |*buf| buf.swap();
            if (self.pass.buffer4) |*buf| buf.swap();
        }

        fn render(self: *@This(), encoder: *gpu.CommandEncoder, screen: *gpu.TextureView) void {
            if (self.pass.buffer1) |*buf| buf.render(encoder, self.channels.view(buf.output));
            if (self.pass.buffer2) |*buf| buf.render(encoder, self.channels.view(buf.output));
            if (self.pass.buffer3) |*buf| buf.render(encoder, self.channels.view(buf.output));
            if (self.pass.buffer4) |*buf| buf.render(encoder, self.channels.view(buf.output));
            self.pass.image.render(encoder, screen);
        }

        fn release(self: *@This()) void {
            self.pass.image.release();
            if (self.pass.buffer1) |*buf| buf.release();
            if (self.pass.buffer2) |*buf| buf.release();
            if (self.pass.buffer3) |*buf| buf.release();
            if (self.pass.buffer4) |*buf| buf.release();
            self.pass.empty_input.tex.release();
            self.pass.empty_input.sampler.release();

            self.channels.release();
            self.binding.deinit(allocator);

            // NOTE: not needed. this is deinited in binding
            // self.uniforms.deinit();
        }
    };

    // gpu has an async api
    // errors callback is called asynchronously. i need a way to to know what shaders and pipelines are good
    // and i don't know of a nice way to do this - so i just say if the gpu does not produce errors for 10
    // frames then the shaders and pipelines are probably good.
    // TODO: could add another fuse to track when to back up current pipelines and bind groups
    //       so that they could be used when new pipelines give errors
    //       (self.err_fuse tracks if error message is printed or not (dumping too many errors is not good))
    const GpuFuse = struct {
        const tick_threshold = 10;
        ticks: std.atomic.Value(u32),
        err_fuse: Fuse,

        fn tick(self: *@This()) void {
            _ = self.ticks.fetchAdd(1, .acq_rel);
            const ticks = self.ticks.fetchMin(tick_threshold, .acq_rel);
            if (ticks >= tick_threshold) {
                _ = self.err_fuse.unfuse();
            }
        }

        fn fuse(self: *@This()) bool {
            self.ticks.store(0, .release);
            return self.err_fuse.fuse();
        }

        fn check(self: *@This()) bool {
            return self.err_fuse.check();
        }
    };

    const Config = struct {
        shader_compile_opt: compile.Glslc.Compiler.Opt = .fast,
        shader_dump_assembly: bool = false,
        pause_cursor: bool = false,
    };

    timer: mach.Timer,
    toyman: Shadertoy.ToyMan,
    pri: PlaygroundRenderInfo,
    gpu_err_fuse: *GpuFuse,
    config: Config = .{},

    fn deinit(self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        self.toyman.deinit();
        self.pri.release();
        allocator.destroy(self.gpu_err_fuse);
    }

    fn init(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        var man = try Shadertoy.ToyMan.init();
        errdefer man.deinit();

        const render_config: Config = .{};

        // TODO: this initialzation should be all from @embed() shaders
        try man.load_zhadertoy("new");
        var pri = try PlaygroundRenderInfo.init(&man.active_toy, &render_config, core_mod);
        errdefer pri.release();

        const core: *mach.Core = core_mod.state();
        const device = core.device;

        const err_fuse = try allocator.create(GpuFuse);

        // - [machengine.org/content/engine/gpu/errors.md](https://github.com/hexops/machengine.org/blob/e10e4245c4c1de7fe9e4882e378a86d42eb1035a/content/engine/gpu/errors.md)
        device.setUncapturedErrorCallback(err_fuse, struct {
            inline fn callback(ctx: *GpuFuse, typ: gpu.ErrorType, message: [*:0]const u8) void {
                switch (typ) {
                    .no_error => return,
                    .validation => {},
                    else => |err| {
                        std.debug.print("Unhandled Gpu Error: {any}\n", .{err});
                        std.debug.print("Error Message: {s}\n", .{message});
                        std.process.exit(1);
                        return;
                    },
                }

                if (!ctx.fuse()) {
                    std.debug.print("Gpu Validation Error: {s}\n", .{message});
                }
            }
        }.callback);

        self_mod.init(.{
            .timer = try mach.Timer.start(),
            .toyman = man,

            .pri = pri,
            .gpu_err_fuse = err_fuse,
            .config = render_config,
        });
    }

    fn render(
        app_mod: *App.Mod,
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
    ) !void {
        const self: *@This() = self_mod.state();
        const core: *mach.Core = core_mod.state();
        const device: *gpu.Device = core.device;
        const app: *App = app_mod.state();

        const label = @tagName(name) ++ ".render";
        const encoder = device.createCommandEncoder(&.{ .label = label });
        defer encoder.release();

        // TODO: more updates left
        self.pri.binding.update(encoder);

        self.pri.swap();
        self.pri.render(encoder, app.rendering_data.?.screen);

        var command = encoder.finish(&.{ .label = label });
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});

        self.gpu_err_fuse.tick();
    }

    fn update_shaders(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const self: *@This() = self_mod.state();

        while (self.toyman.try_get_update()) |ev| {
            std.debug.print("updating shader: {any}\n", .{ev});

            self.pri.update(&self.toyman.active_toy, &self.config, core_mod, ev) catch |e| {
                std.debug.print("Error while updating shaders: {any}\n", .{e});
            };
            _ = self.gpu_err_fuse.err_fuse.unfuse();
        }
    }

    fn tick(self_mod: *Mod, app_mod: *App.Mod, core_mod: *mach.Core.Mod) !void {
        const app: *App = app_mod.state();

        const self: *@This() = self_mod.state();
        if (self.toyman.has_updates()) {
            self_mod.schedule(.update_shaders);
        }

        var state = &self.pri.uniforms.uniforms.val;
        state.time_delta = self.timer.lap();
        state.time += state.time_delta;
        state.frame += 1;

        for (app.events.items) |event| {
            switch (event) {
                .mouse_motion => |pos| {
                    if (!self.config.pause_cursor) {
                        state.mouse_x = @floatCast(pos.pos.x);
                        state.mouse_y = @floatCast(pos.pos.y);
                    }
                },
                .mouse_press => |button| {
                    switch (button.button) {
                        .left => {
                            // state.mouse_left = 1;
                        },
                        .right => {
                            // state.mouse_right = 1;
                        },
                        .middle => {
                            // state.mouse_middle = 1;
                        },
                        else => {},
                    }
                },
                .mouse_release => |button| {
                    switch (button.button) {
                        .left => {
                            // state.mouse_left = 0;
                        },
                        .right => {
                            // state.mouse_right = 0;
                        },
                        .middle => {
                            // state.mouse_middle = 0;
                        },
                        else => {},
                    }
                },
                .mouse_scroll => |del| {
                    _ = del;
                    // state.scroll += del.yoffset;
                },
                .key_press => |ev| {
                    switch (ev.key) {
                        .escape => {
                            core_mod.schedule(.exit);
                        },
                        else => {},
                    }
                },
                .key_release => |ev| {
                    switch (ev.key) {
                        else => {},
                    }
                },
                .framebuffer_resize => |sze| {
                    state.width = sze.width;
                    state.height = sze.height;
                },
                else => {},
            }
        }
    }
};

const App = struct {
    pub const name = .app;
    pub const systems = .{
        .init = .{ .handler = init },
        .deinit = .{ .handler = deinit },
        .tick = .{ .handler = tick },
        .render = .{ .handler = render },
        .present_frame = .{ .handler = present_frame },
    };
    pub const Mod = mach.Mod(@This());

    events: std.ArrayList(mach.core.Event),
    rendering_data: ?struct {
        screen: *gpu.TextureView,
    } = null,

    fn init(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
        renderer_mod: *Renderer.Mod,
        gui_mod: *Gui.Mod,
    ) !void {
        core_mod.schedule(.init);
        defer core_mod.schedule(.start);

        // NOTE: initialize after core
        gui_mod.schedule(.init);
        renderer_mod.schedule(.init);

        self_mod.init(.{
            .events = std.ArrayList(mach.core.Event).init(allocator),
        });
    }

    fn deinit(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
        renderer_mod: *Renderer.Mod,
        gui_mod: *Gui.Mod,
    ) !void {
        const self: *@This() = self_mod.state();

        gui_mod.schedule(.deinit);
        renderer_mod.schedule(.deinit);
        core_mod.schedule(.deinit);

        if (self.rendering_data) |data| data.screen.release();
        self.rendering_data = null;
        self.events.deinit();
    }

    fn tick(
        self_mod: *Mod,
        renderer_mod: *Renderer.Mod,
        core_mod: *mach.Core.Mod,
        gui_mod: *Gui.Mod,
    ) !void {
        const self: *@This() = self_mod.state();
        renderer_mod.schedule(.tick);
        gui_mod.schedule(.tick);

        defer self_mod.schedule(.render);

        self.events.clearRetainingCapacity();

        var iter = mach.core.pollEvents();
        while (iter.next()) |event| {
            try self.events.append(event);
            switch (event) {
                .close => core_mod.schedule(.exit),
                else => {},
            }
        }
    }

    fn render(
        self_mod: *Mod,
        renderer_mod: *Renderer.Mod,
        gui_mod: *Gui.Mod,
    ) void {
        const self: *@This() = self_mod.state();
        self.rendering_data = .{
            .screen = mach.core.swap_chain.getCurrentTextureView().?,
        };

        renderer_mod.schedule(.render);
        gui_mod.schedule(.render);

        self_mod.schedule(.present_frame);
    }

    fn present_frame(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
    ) void {
        const self: *@This() = self_mod.state();

        self.rendering_data.?.screen.release();
        self.rendering_data = null;

        core_mod.schedule(.present_frame);
    }
};

const Gui = struct {
    const imgui = @import("imgui");
    const imgui_mach = imgui.backends.mach;

    var allocator_imgui = allocator;

    pub const name = .gui;
    pub const systems = .{
        .init = .{ .handler = init },
        .deinit = .{ .handler = deinit },
        .tick = .{ .handler = tick },
        .render = .{ .handler = render },
    };
    pub const Mod = mach.Mod(@This());

    title_timer: mach.Timer,

    fn init(self_mod: *Mod, core_mod: *mach.Core.Mod) !void {
        const core: *mach.Core = core_mod.state();

        imgui.setZigAllocator(&allocator_imgui);
        _ = imgui.createContext(null);
        try imgui_mach.init(allocator, core.device, .{});

        var io = imgui.getIO();
        io.config_flags |= imgui.ConfigFlags_NavEnableKeyboard;
        io.font_global_scale = 1.0 / io.display_framebuffer_scale.y;

        const style = imgui.getStyle();
        style.colors[imgui.Col_Text] = imgui.Vec4{ .x = 0.93, .y = 0.93, .z = 0.93, .w = 1.00 };
        style.colors[imgui.Col_TextDisabled] = imgui.Vec4{ .x = 0.5, .y = 0.5, .z = 0.5, .w = 1.00 };
        style.colors[imgui.Col_WindowBg] = imgui.Vec4{ .x = 0.11, .y = 0.11, .z = 0.11, .w = 1.00 };
        style.colors[imgui.Col_ChildBg] = imgui.Vec4{ .x = 0.15, .y = 0.15, .z = 0.15, .w = 1.00 };
        style.colors[imgui.Col_Border] = imgui.Vec4{ .x = 0.30, .y = 0.30, .z = 0.30, .w = 1.00 };
        style.colors[imgui.Col_FrameBg] = imgui.Vec4{ .x = 0.20, .y = 0.20, .z = 0.20, .w = 1.00 };
        style.colors[imgui.Col_FrameBgHovered] = imgui.Vec4{ .x = 0.40, .y = 0.40, .z = 0.40, .w = 1.00 };
        style.colors[imgui.Col_FrameBgActive] = imgui.Vec4{ .x = 0.50, .y = 0.50, .z = 0.50, .w = 1.00 };
        style.colors[imgui.Col_TitleBg] = imgui.Vec4{ .x = 0.00, .y = 0.00, .z = 0.00, .w = 1.00 };
        style.colors[imgui.Col_TitleBgActive] = imgui.Vec4{ .x = 0.20, .y = 0.20, .z = 0.20, .w = 1.00 };
        style.colors[imgui.Col_TitleBgCollapsed] = imgui.Vec4{ .x = 0.10, .y = 0.10, .z = 0.10, .w = 1.00 };
        style.colors[imgui.Col_Button] = imgui.Vec4{ .x = 0.80, .y = 0.20, .z = 0.20, .w = 1.00 };
        style.colors[imgui.Col_ButtonHovered] = imgui.Vec4{ .x = 1.00, .y = 0.50, .z = 0.50, .w = 1.00 };
        style.colors[imgui.Col_ButtonActive] = imgui.Vec4{ .x = 1.00, .y = 0.30, .z = 0.30, .w = 1.00 };
        style.window_rounding = 5;
        style.frame_rounding = 3;

        // const font_data = @embedFile("Roboto-Medium.ttf");
        // const size_pixels = 12 * io.display_framebuffer_scale.y;

        // var font_cfg: imgui.FontConfig = std.mem.zeroes(imgui.FontConfig);
        // font_cfg.font_data_owned_by_atlas = false;
        // font_cfg.oversample_h = 2;
        // font_cfg.oversample_v = 1;
        // font_cfg.glyph_max_advance_x = std.math.floatMax(f32);
        // font_cfg.rasterizer_multiply = 1.0;
        // font_cfg.rasterizer_density = 1.0;
        // font_cfg.ellipsis_char = imgui.UNICODE_CODEPOINT_MAX;
        // _ = io.fonts.?.addFontFromMemoryTTF(@constCast(@ptrCast(font_data.ptr)), font_data.len, size_pixels, &font_cfg, null);

        self_mod.init(.{
            .title_timer = try mach.Timer.start(),
        });
    }

    fn deinit() void {
        imgui_mach.shutdown();
        imgui.destroyContext(null);
    }

    fn tick(app_mod: *App.Mod, self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        const app: *App = app_mod.state();

        for (app.events.items) |event| {
            _ = imgui_mach.processEvent(event);
        }

        // update the window title every second
        if (self.title_timer.read() >= 1.0) {
            self.title_timer.reset();
            try mach.core.printTitle("ImGui [ {d}fps ] [ Input {d}hz ]", .{
                mach.core.frameRate(),
                mach.core.inputRate(),
            });
        }
    }

    fn render(app_mod: *App.Mod, renderer_mod: *Renderer.Mod, core_mod: *mach.Core.Mod) !void {
        const core: *mach.Core = core_mod.state();
        const app: *App = app_mod.state();
        const renderer: *Renderer = renderer_mod.state();

        imgui_mach.newFrame() catch return;
        imgui.newFrame();

        const io = imgui.getIO();

        {
            defer imgui.render();

            imgui.setNextWindowPos(.{ .x = 5, .y = 5 }, imgui.Cond_Once);
            // imgui.setNextWindowSize(.{ .x = 400, .y = 300 }, imgui.Cond_Once);
            // imgui.setNextWindowCollapsed(true, imgui.Cond_Once);
            defer imgui.end();
            if (imgui.begin("SIKE!", null, imgui.WindowFlags_None)) {
                imgui.text("Application average %.3f ms/frame (%.1f FPS)", 1000.0 / io.framerate, io.framerate);
                _ = imgui.checkbox("pause cursor", &renderer.config.pause_cursor);
                _ = imgui.checkbox("shader dump assembly", &renderer.config.shader_dump_assembly);

                enum_checkbox(&renderer.config.shader_compile_opt, "shader compile opt mode");
                pick_toy(renderer) catch |e| std.debug.print("{any}\n", .{e});
            }

            // imgui.showDemoWindow(null);
        }

        const color_attachment = gpu.RenderPassColorAttachment{
            .view = app.rendering_data.?.screen,
            .clear_value = gpu.Color{
                .r = 40.0 / 255.0,
                .g = 40.0 / 255.0,
                .b = 40.0 / 255.0,
                .a = 1,
            },
            .load_op = .load,
            .store_op = .store,
        };

        const encoder = core.device.createCommandEncoder(null);
        defer encoder.release();

        const render_pass_info = gpu.RenderPassDescriptor.init(.{
            .color_attachments = &.{color_attachment},
        });
        const pass = encoder.beginRenderPass(&render_pass_info);
        defer pass.release();

        imgui_mach.renderDrawData(imgui.getDrawData().?, pass) catch {};
        pass.end();

        var command = encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }

    var pick_toy_index: c_int = 0;
    fn pick_toy(renderer: *Renderer) !void {
        const toys = [_][*:0]const u8{
            "z:new: new yo",
            "z:nonuniform_control_flow: broken",
            "s:4td3zj: curly stuff",
            "s:lXjyWt: neon blob",
            "s:lX2yDt: earth",
            "s:NslGRN: cool glass cube",
            "s:4ttSWf: iq forest",
            "s:MdX3Rr: iq 3d noise mountain",
            "s:WlKXzm: satellite thing",
            "s:43lfRB: cool blue 3d fractal",
        };
        imgui.setNextItemWidth(200);
        _ = imgui.comboChar("shadertoy", &pick_toy_index, &toys, toys.len);
        imgui.sameLine();
        const toy = toys[@intCast(pick_toy_index)];
        if (imgui.button("reload")) {
            const id = std.mem.span(@as([*:':']const u8, @ptrCast(toy[2..])));
            switch (toy[0]) {
                's' => {
                    try renderer.toyman.load_shadertoy(id);
                },
                'z' => {
                    try renderer.toyman.load_zhadertoy(id);
                },
                else => return error.UnknownTypeOfToy,
            }
        }
    }

    fn enum_checkbox(enum_ptr: anytype, title: [*:0]const u8) void {
        const opt_modes = comptime blk: {
            const fields = @typeInfo(@TypeOf(enum_ptr.*)).Enum.fields;
            var arr: [fields.len][*:0]const u8 = undefined;
            for (fields, 0..) |field, i| {
                arr[i] = field.name;
            }
            break :blk arr;
        };
        var opt_index: c_int = 0;
        for (opt_modes, 0..) |mode, i| {
            if (std.mem.eql(u8, std.mem.span(mode), @tagName(enum_ptr.*))) {
                opt_index = @intCast(i);
            }
        }
        _ = imgui.comboChar(title, &opt_index, &opt_modes, opt_modes.len);
        enum_ptr.* = std.meta.stringToEnum(@TypeOf(enum_ptr.*), std.mem.span(opt_modes[@intCast(opt_index)])).?;
    }
};

pub fn main() !void {
    // try Glslang.testfn();
    // try Dawn.testfn();
    // try Shaderc.testfn();
    // try Glslc.testfn();
    // try Shadertoy.testfn();
    // try FsFuse.testfn();
    // if (true) {
    //     return;
    // }

    try mach.core.initModule();
    while (try mach.core.tick()) {}

    // no defer cuz we don't want to print leaks when we error out
    _ = gpa.deinit();
}
