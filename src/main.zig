const std = @import("std");

const compile = @import("compile.zig");
const Glslc = compile.Glslc;

const Shadertoy = @import("shadertoy.zig");

const mach = @import("mach");

// using wgpu instead of mach.gpu for better lsp results
const gpu = mach.wgpu;

// The global list of Mach modules registered for use in our application.
pub const modules = .{
    mach.Core,
    App,
    Renderer,
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
        .render_frame = .{ .handler = render_frame },
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

    const Vec3 = extern struct {
        x: f32,
        y: f32,
        z: f32,
        w: f32 = 0.0,
    };
    const ShadertoyUniforms = struct {
        time: *Buffer(f32),
        resolution: *Buffer(Vec3),

        fn init(core_mod: *mach.Core.Mod) !@This() {
            const core: *mach.Core = core_mod.state();
            const device = core.device;
            const height = core_mod.get(core.main_window, .height).?;
            const width = core_mod.get(core.main_window, .width).?;
            var time = try Buffer(f32).new(@tagName(name) ++ " time", 0.0, allocator, device);
            errdefer time.deinit(allocator);
            return .{
                .time = time,
                .resolution = try Buffer(Vec3).new(@tagName(name) ++ " resolution", Vec3{
                    .x = @floatFromInt(width),
                    .y = @floatFromInt(height),
                    .z = 0.0,
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
                core_mod: *mach.Core.Mod,
                channels: *Channels,
                binding: *Binding,
                empty_input: *EmptyInput,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var inputs = Pass.Inputs.init(device, &at.passes.image.inputs);
                errdefer inputs.release();
                var bind = try Pass.create_bind_group(device, binding, channels, &inputs, empty_input);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try Pass.create_pipeline(device, &inputs, at.has_common, bind.layout, core_mod.get(core.main_window, .framebuffer_format).?, .image);
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
                var bind = try create_bind_group(device, binding, channels, &inputs, empty_input);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try create_pipeline(device, &inputs, at.has_common, bind.layout, .rgba32_float, switch (include) {
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

                    fn add_all(self: *@This(), channels: *Channels, inputs: *Inputs, current: bool) !void {
                        if (inputs.input1) |inp| {
                            const tex = channels.get(inp.typ);
                            const view = if (current) tex.current.view else tex.last_frame.view;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input2) |inp| {
                            const tex = channels.get(inp.typ);
                            const view = if (current) tex.current.view else tex.last_frame.view;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input3) |inp| {
                            const tex = channels.get(inp.typ);
                            const view = if (current) tex.current.view else tex.last_frame.view;
                            try self.add(view, inp.sampler);
                        } else {
                            try self.add(self.empty.tex.view, self.empty.sampler);
                        }
                        if (inputs.input4) |inp| {
                            const tex = channels.get(inp.typ);
                            const view = if (current) tex.current.view else tex.last_frame.view;
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
                try fnn1.add_all(_channels, _inputs, true);

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
                try fnn2.add_all(_channels, _inputs, false);

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

                var compiler = Glslc.Compiler{ .opt = .fast };
                compiler.stage = .vertex;
                // _ = compiler.dump_assembly(allocator, vert);
                const vertex_bytes = try compiler.compile(
                    allocator,
                    &.{ .path = .{
                        .main = "./shaders/vert.glsl",
                        .include = &[_][]const u8{config.paths.playground},
                        .definitions = definitions.items,
                    } },
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
                // _ = compiler.dump_assembly(allocator, frag);
                const frag_bytes = try compiler.compile(
                    allocator,
                    &.{ .path = .{
                        .main = "./shaders/frag.glsl",
                        .include = &[_][]const u8{config.paths.playground},
                        .definitions = definitions.items,
                    } },
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
                render_pass.setBindGroup(0, bg, &.{ 0, 0 });
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
        uniforms: ShadertoyUniforms,

        fn init(at: *Shadertoy.ActiveToy, core_mod: *mach.Core.Mod) !@This() {
            const core: *mach.Core = core_mod.state();
            const device: *gpu.Device = core.device;

            const uniforms = try ShadertoyUniforms.init(core_mod);

            var binding = Binding{ .label = "fajdl;f;jda" };
            errdefer binding.deinit(allocator);
            // LEAK: if .add() errors, those uniforms don't get deinited
            try binding.add(uniforms.time, allocator);
            try binding.add(uniforms.resolution, allocator);
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
            var image_pass = try ImagePass.init(at, core_mod, &channels, &binding, &empty_input);
            errdefer image_pass.release();
            var pass1 = if (at.passes.buffer1) |*pass| try Pass.init(at, core_mod, pass, &channels, &binding, .buffer1, &empty_input) else null;
            errdefer if (pass1) |*pass| pass.release();
            var pass2 = if (at.passes.buffer2) |*pass| try Pass.init(at, core_mod, pass, &channels, &binding, .buffer2, &empty_input) else null;
            errdefer if (pass2) |*pass| pass.release();
            var pass3 = if (at.passes.buffer3) |*pass| try Pass.init(at, core_mod, pass, &channels, &binding, .buffer3, &empty_input) else null;
            errdefer if (pass3) |*pass| pass.release();
            var pass4 = if (at.passes.buffer4) |*pass| try Pass.init(at, core_mod, pass, &channels, &binding, .buffer4, &empty_input) else null;
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

        fn update(self: *@This(), at: *Shadertoy.ActiveToy, core_mod: *mach.Core.Mod, ev: Shadertoy.ToyMan.UpdateEvent) !void {
            switch (ev) {
                .Buffer1 => {
                    var buf1 = at.passes.buffer1 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer1.?;
                    const new_pass = try Pass.init(at, core_mod, &buf1, &self.channels, &self.binding, .buffer1, &self.pass.empty_input);
                    self.pass.buffer1 = new_pass;
                    pass.release();
                },
                .Buffer2 => {
                    var buf2 = at.passes.buffer2 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer2.?;
                    const new_pass = try Pass.init(at, core_mod, &buf2, &self.channels, &self.binding, .buffer2, &self.pass.empty_input);
                    self.pass.buffer2 = new_pass;
                    pass.release();
                },
                .Buffer3 => {
                    var buf3 = at.passes.buffer3 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer3.?;
                    const new_pass = try Pass.init(at, core_mod, &buf3, &self.channels, &self.binding, .buffer3, &self.pass.empty_input);
                    self.pass.buffer3 = new_pass;
                    pass.release();
                },
                .Buffer4 => {
                    var buf4 = at.passes.buffer4 orelse return error.BufferPassDoesNotExist;
                    var pass = self.pass.buffer4.?;
                    const new_pass = try Pass.init(at, core_mod, &buf4, &self.channels, &self.binding, .buffer4, &self.pass.empty_input);
                    self.pass.buffer4 = new_pass;
                    pass.release();
                },
                .Image => {
                    var pass = self.pass.image;
                    const new_pass = try ImagePass.init(at, core_mod, &self.channels, &self.binding, &self.pass.empty_input);
                    self.pass.image = new_pass;
                    pass.release();
                },
                .All => {
                    self.release();
                    self.* = try @This().init(at, core_mod);
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

    timer: mach.Timer,
    toyman: Shadertoy.ToyMan,
    pri: PlaygroundRenderInfo,

    fn deinit(self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        self.toyman.deinit();
        self.pri.release();
    }

    fn init(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        var man = try Shadertoy.ToyMan.init();
        errdefer man.deinit();

        // TODO: this initialzation should be all from @embed() shaders
        try man.load_zhadertoy("new");
        var pri = try PlaygroundRenderInfo.init(&man.active_toy, core_mod);
        errdefer pri.release();

        self_mod.init(.{
            .timer = try mach.Timer.start(),
            .toyman = man,

            .pri = pri,
        });
    }

    fn render_frame(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
    ) !void {
        const self: *@This() = self_mod.state();
        const core: *mach.Core = core_mod.state();
        const device: *gpu.Device = core.device;

        const label = @tagName(name) ++ ".render_frame";
        const encoder = device.createCommandEncoder(&.{ .label = label });
        defer encoder.release();

        // TODO: more updates left
        self.pri.binding.update(encoder);

        const back_buffer_view = mach.core.swap_chain.getCurrentTextureView().?;
        defer back_buffer_view.release();

        self.pri.swap();
        self.pri.render(encoder, back_buffer_view);

        var command = encoder.finish(&.{ .label = label });
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});

        core_mod.schedule(.present_frame);
    }

    fn update_shaders(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const self: *@This() = self_mod.state();

        while (self.toyman.try_get_update()) |ev| {
            self.pri.update(&self.toyman.active_toy, core_mod, ev) catch |e| {
                std.debug.print("Error while updating shaders: {any}\n", .{e});
            };
        }
    }

    fn tick(self_mod: *Mod, core_mod: *mach.Core.Mod) !void {
        defer self_mod.schedule(.render_frame);

        const self: *@This() = self_mod.state();
        if (self.toyman.has_updates()) {
            self_mod.schedule(.update_shaders);
        }

        var state = &self.pri.uniforms;
        state.time.val += self.timer.lap();

        // TODO: shadertoy uniforms
        var iter = mach.core.pollEvents();
        while (iter.next()) |event| {
            switch (event) {
                .mouse_motion => |pos| {
                    _ = pos;
                    // state.cursor_x = @floatCast(pos.pos.x);
                    // state.cursor_y = @floatCast(pos.pos.y);
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
                        .one => {
                            try self.toyman.load_shadertoy("4td3zj");
                        },
                        .two => {
                            // try self.toyman.load_shadertoy("lXjyWt");
                            try self.toyman.load_shadertoy("lX2yDt");
                        },
                        .zero => {
                            try self.toyman.load_zhadertoy("new");
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
                    _ = sze;
                    // state.render_height = sze.height;
                    // state.render_width = sze.width;
                    // state.display_height = sze.height;
                    // state.display_width = sze.width;
                },
                .close => core_mod.schedule(.exit),
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
    };
    pub const Mod = mach.Mod(@This());

    fn init(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
        renderer_mod: *Renderer.Mod,
    ) !void {
        core_mod.schedule(.init);
        defer core_mod.schedule(.start);

        // NOTE: core should be initialized before renderer
        renderer_mod.schedule(.init);

        self_mod.init(.{});
    }

    fn deinit(core_mod: *mach.Core.Mod, renderer_mod: *Renderer.Mod) !void {
        renderer_mod.schedule(.deinit);
        core_mod.schedule(.deinit);
    }

    fn tick(
        renderer_mod: *Renderer.Mod,
    ) !void {
        renderer_mod.schedule(.tick);

        // NOOOO: can't poll events in multiple systems (it consumes)
        // var iter = mach.core.pollEvents();
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
