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

pub const Renderer = struct {
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
        _pad: u32 = 0,

        resolution: Vec3,
        _pad2: u32 = 0,

        mouse: Vec4 = .{},

        width: u32,
        height: u32,
        mouse_x: f32 = 0.0,
        mouse_y: f32 = 0.0,

        mouse_left: bool = false,
        mouse_right: bool = false,
        mouse_middle: bool = false,

        fn reset_state(self: *@This()) void {
            self.time = 0.0;
            self.time_delta = 0.0;
            self.frame = 0;
            self.mouse_x = 0.0;
            self.mouse_y = 0.0;
            self.mouse_left = false;
            self.mouse_right = false;
            self.mouse_middle = false;
            self.mouse = .{};
        }
    };
    const ShadertoyUniformBuffers = struct {
        uniforms: *Buffer(ShadertoyUniforms),

        fn init(device: *gpu.Device, height: u32, width: u32) !@This() {
            return .{
                .uniforms = try Buffer(ShadertoyUniforms).new(@tagName(name) ++ " uniforms", .{
                    .width = width,
                    .height = height,
                    .resolution = .{ .x = @floatFromInt(width), .y = @floatFromInt(height) },
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
        const ScreenPass = struct {
            pipeline: *gpu.RenderPipeline,
            bind_group: *gpu.BindGroup,
            bind_group_layout: *gpu.BindGroupLayout,

            fn init(
                core_mod: *mach.Core.Mod,
                binding: *Binding,
                view: *gpu.TextureView,
                sampler: *gpu.Sampler,
                shader: *const Shadertoy.ToyMan.CompiledShader,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var bind = try create_bind_group(device, binding, view, sampler);
                errdefer {
                    bind.group.release();
                    bind.layout.release();
                }
                var pipeline = try Pass.create_pipeline(device, bind.layout, core_mod.get(core.main_window, .framebuffer_format).?, shader);
                errdefer pipeline.release();

                return .{
                    .bind_group_layout = bind.layout,
                    .bind_group = bind.group,
                    .pipeline = pipeline,
                };
            }

            fn create_bind_group(
                device: *gpu.Device,
                binding: *Binding,
                view: *gpu.TextureView,
                sampler: *gpu.Sampler,
            ) !struct { group: *gpu.BindGroup, layout: *gpu.BindGroupLayout } {
                const alloc = allocator;
                var layout_entries = std.ArrayListUnmanaged(gpu.BindGroupLayout.Entry){};
                defer layout_entries.deinit(alloc);
                var bind_group_entries = std.ArrayListUnmanaged(gpu.BindGroup.Entry){};
                defer bind_group_entries.deinit(alloc);

                for (binding.buffers.items, 0..) |*buf, i| {
                    try layout_entries.append(alloc, buf.bindGroupLayoutEntry(@intCast(i)));

                    try bind_group_entries.append(alloc, buf.bindGroupEntry(@intCast(i)));
                }

                const visibility = .{
                    .vertex = true,
                    .fragment = true,
                    .compute = true,
                };

                try layout_entries.append(alloc, gpu.BindGroupLayout.Entry.texture(
                    @intCast(layout_entries.items.len),
                    visibility,
                    .float,
                    .dimension_2d,
                    false,
                ));
                try bind_group_entries.append(alloc, gpu.BindGroup.Entry.textureView(@intCast(bind_group_entries.items.len), view));
                try layout_entries.append(alloc, gpu.BindGroupLayout.Entry.sampler(
                    @intCast(layout_entries.items.len),
                    visibility,
                    .filtering,
                ));
                try bind_group_entries.append(alloc, gpu.BindGroup.Entry.sampler(@intCast(bind_group_entries.items.len), sampler));

                const bind_group_layout = device.createBindGroupLayout(
                    &gpu.BindGroupLayout.Descriptor.init(.{
                        .label = "screen pass bind group",
                        .entries = layout_entries.items,
                    }),
                );
                const bind_group = device.createBindGroup(
                    &gpu.BindGroup.Descriptor.init(.{
                        .label = "bind grpup adfasf",
                        .layout = bind_group_layout,
                        .entries = bind_group_entries.items,
                    }),
                );

                return .{
                    .group = bind_group,
                    .layout = bind_group_layout,
                };
            }

            fn render(self: *@This(), encoder: *gpu.CommandEncoder, screen: *gpu.TextureView) void {
                Pass._render_pass(self.pipeline, self.bind_group, encoder, screen);
            }

            fn release(self: *@This()) void {
                self.bind_group.release();
                self.bind_group_layout.release();
                self.pipeline.release();
            }
        };
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
                channels: *Buffers,
                binding: *Binding,
                shader: *const Shadertoy.ToyMan.CompiledShader,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var inputs = Pass.Inputs.init(device, &at.passes.image.inputs);
                errdefer inputs.release();
                var bind = try Pass.create_bind_group(device, binding, channels, &inputs);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try Pass.create_pipeline(device, bind.layout, .rgba32_float, shader);
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
                self.bind_group_layout.release();
                self.pipeline.release();
            }
        };
        const Pass = struct {
            const Input = struct {
                // OOF: FIXME: typ is not owned. but there's no indication/care given about it.
                //  maybe just store a key into a hashmap stored in ActiveToy to avoid big 'typ'
                typ: Shadertoy.ActiveToy.Buffer,
                sampler: *gpu.Sampler,
                info: Shadertoy.ActiveToy.Input,

                fn init(device: *gpu.Device, input: ?Shadertoy.ActiveToy.Input) ?@This() {
                    if (input) |inp| {
                        const wrap: gpu.Sampler.AddressMode = switch (inp.sampler.wrap) {
                            .clamp => .clamp_to_edge,
                            .repeat => .repeat,
                        };
                        const filter: gpu.FilterMode = switch (inp.sampler.filter) {
                            .nearest => .nearest,
                            .linear => .linear,
                            .mipmap => .linear,
                        };
                        const mipmap_filter: gpu.MipmapFilterMode = switch (inp.sampler.filter) {
                            .nearest => .nearest,
                            .linear => .linear,
                            .mipmap => .linear,
                        };
                        return .{
                            .typ = inp.typ,
                            .sampler = device.createSampler(&.{
                                .address_mode_u = wrap,
                                .address_mode_v = wrap,
                                .address_mode_w = wrap,
                                .mag_filter = filter,
                                .min_filter = filter,
                                .mipmap_filter = mipmap_filter,
                            }),
                            .info = inp,
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

                fn init(device: *gpu.Device, inputs: *const Shadertoy.ActiveToy.Inputs) @This() {
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
                core_mod: *mach.Core.Mod,
                pass: *Shadertoy.ActiveToy.BufferPass,
                channels: *Buffers,
                binding: *Binding,
                shader: *const Shadertoy.ToyMan.CompiledShader,
            ) !@This() {
                const core: *mach.Core = core_mod.state();
                const device = core.device;

                var inputs = Inputs.init(device, &pass.inputs);
                var bind = try create_bind_group(device, binding, channels, &inputs);
                errdefer {
                    bind.group1.release();
                    bind.group2.release();
                    bind.layout.release();
                }
                var pipeline = try create_pipeline(device, bind.layout, .rgba32_float, shader);
                errdefer pipeline.release();

                return .{
                    .inputs = inputs,
                    .output = pass.output,
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
                _buffers: *Buffers,
                _inputs: *Inputs,
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
                    alloc: @TypeOf(alloc),

                    const visibility = .{
                        .vertex = true,
                        .fragment = true,
                        .compute = true,
                    };

                    fn add(self: *@This(), view: *gpu.TextureView, sampler: *gpu.Sampler, info: ?Shadertoy.ActiveToy.Input) !void {
                        try self.layouts.append(self.alloc, gpu.BindGroupLayout.Entry.texture(
                            @intCast(self.layouts.items.len),
                            visibility,
                            if (info) |i| switch (i.sampler.filter) {
                                .nearest => .unfilterable_float,
                                .linear => .float,
                                .mipmap => .float,
                            } else .unfilterable_float,
                            .dimension_2d,
                            false,
                        ));
                        try self.groups.append(self.alloc, gpu.BindGroup.Entry.textureView(@intCast(self.groups.items.len), view));
                        try self.layouts.append(self.alloc, gpu.BindGroupLayout.Entry.sampler(
                            @intCast(self.layouts.items.len),
                            visibility,
                            if (info) |i| switch (i.sampler.filter) {
                                .nearest => .non_filtering,
                                .linear => .filtering,
                                .mipmap => .filtering,
                            } else .non_filtering,
                        ));
                        try self.groups.append(self.alloc, gpu.BindGroup.Entry.sampler(@intCast(self.groups.items.len), sampler));
                    }

                    fn add_all(self: *@This(), buffers: *Buffers, inputs: *Inputs, swap_tex: bool) !void {
                        if (inputs.input1) |inp| {
                            const tex = if (swap_tex != inp.info.from_current_frame) buffers.get_current(inp.typ) else buffers.get_last_frame(inp.typ);
                            try self.add(tex.view, inp.sampler, inp.info);
                        } else {
                            try self.add(buffers.empty_input.tex.view, buffers.empty_input.sampler, null);
                        }
                        if (inputs.input2) |inp| {
                            const tex = if (swap_tex != inp.info.from_current_frame) buffers.get_current(inp.typ) else buffers.get_last_frame(inp.typ);
                            try self.add(tex.view, inp.sampler, inp.info);
                        } else {
                            try self.add(buffers.empty_input.tex.view, buffers.empty_input.sampler, null);
                        }
                        if (inputs.input3) |inp| {
                            const tex = if (swap_tex != inp.info.from_current_frame) buffers.get_current(inp.typ) else buffers.get_last_frame(inp.typ);
                            try self.add(tex.view, inp.sampler, inp.info);
                        } else {
                            try self.add(buffers.empty_input.tex.view, buffers.empty_input.sampler, null);
                        }
                        if (inputs.input4) |inp| {
                            const tex = if (swap_tex != inp.info.from_current_frame) buffers.get_current(inp.typ) else buffers.get_last_frame(inp.typ);
                            try self.add(tex.view, inp.sampler, inp.info);
                        } else {
                            try self.add(buffers.empty_input.tex.view, buffers.empty_input.sampler, null);
                        }
                    }
                };

                var fnn1 = Fn{
                    .layouts = try layout_entries.clone(allocator),
                    .groups = try bind_group_entries.clone(allocator),
                    .alloc = alloc,
                };
                defer {
                    fnn1.layouts.deinit(alloc);
                    fnn1.groups.deinit(alloc);
                }
                try fnn1.add_all(_buffers, _inputs, false);

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
                    .alloc = alloc,
                };
                defer {
                    fnn2.layouts.deinit(alloc);
                    fnn2.groups.deinit(alloc);
                }
                try fnn2.add_all(_buffers, _inputs, true);

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
                bg_layout: *gpu.BindGroupLayout,
                format: gpu.Texture.Format,
                shader: *const Shadertoy.ToyMan.CompiledShader,
            ) !*gpu.RenderPipeline {
                const vertex_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
                    .next_in_chain = .{ .spirv_descriptor = &.{
                        .code_size = @intCast(shader.vert.len),
                        .code = shader.vert.ptr,
                    } },
                    .label = "vert.glsl",
                });
                defer vertex_shader_module.release();
                const frag_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
                    .next_in_chain = .{
                        .spirv_descriptor = &.{
                            .code_size = @intCast(shader.frag.len),
                            .code = shader.frag.ptr,
                            .chain = .{
                                // - [WebGPU Shading Language](https://www.w3.org/TR/WGSL/#uniformity)
                                // - [Designing a uniformity opt-out (similar thing exists in dawn :})](https://github.com/gpuweb/gpuweb/issues/3554)
                                .next = @ptrCast(&gpu.dawn.ShaderModuleSPIRVOptionsDescriptor{
                                    .allow_non_uniform_derivatives = .true,
                                }),
                                .s_type = .shader_module_spirv_descriptor,
                            },
                        },
                    },
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
                    .clear_value = utils.ColorParse.hex_rgba(gpu.Color, "#282828ff"),
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
                self.bind_group_layout.release();
                self.pipeline.release();
            }
        };
        const Channel = struct {
            const Tex = struct {
                texture: *gpu.Texture,
                view: *gpu.TextureView,

                fn init(device: *gpu.Device, size: gpu.Extent3D, format: gpu.Texture.Format, comptime label: [:0]const u8) @This() {
                    const tex_desc = gpu.Texture.Descriptor.init(.{
                        .label = "texture " ++ label,
                        .size = size,
                        .dimension = .dimension_2d,
                        .usage = .{
                            .copy_src = true,
                            .texture_binding = true,
                            .render_attachment = true,
                        },
                        .format = format,
                        // .view_format = &[_]gpu.Texture.Format{ .rgba8_unorm, .rgba32_float },
                    });
                    const tex = device.createTexture(&tex_desc);
                    return .{
                        .texture = tex,
                        .view = tex.createView(&.{}),
                        // .view = tex.createView(&.{ .format = .rgba16_float }),
                    };
                }

                fn from_img(device: *gpu.Device, queue: *gpu.Queue, img: *utils.ImageMagick.FloatImage, label: [:0]const u8) @This() {
                    const size = gpu.Extent3D{
                        .width = @intCast(img.width),
                        .height = @intCast(img.height),
                    };
                    const tex_desc = gpu.Texture.Descriptor.init(.{
                        .label = label,
                        .size = size,
                        .dimension = .dimension_2d,
                        .usage = .{
                            .copy_dst = true,
                            .texture_binding = true,
                        },
                        .format = .rgba32_float,
                        // .view_formats = &[_]gpu.Texture.Format{ .rgba8_unorm, .rgba16_float },
                    });
                    const tex = device.createTexture(&tex_desc);

                    // TODO: vflip
                    // - can handle vflip for everything just using shaders ig :/
                    //    - create a new struct (custom type). override all texture functions
                    //      and pass vflip as uniform
                    queue.writeTexture(&.{
                        .texture = tex,
                    }, &.{
                        .bytes_per_row = @sizeOf(utils.ImageMagick.Pixel(f32)) * size.width,
                        .rows_per_image = size.height,
                    }, &size, img.buffer);

                    return .{
                        .texture = tex,
                        .view = tex.createView(&.{}),
                    };
                }

                fn release(self: *@This()) void {
                    self.view.release();
                    self.texture.destroy();
                }
            };
            current: Tex,
            last_frame: Tex,

            fn init(device: *gpu.Device, size: gpu.Extent3D, format: gpu.Texture.Format, comptime label: [:0]const u8) @This() {
                return .{
                    .current = Tex.init(device, size, format, "current " ++ label),
                    .last_frame = Tex.init(device, size, format, "last frame " ++ label),
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
        const Buffers = struct {
            const Sampled = struct {
                tex: Channel.Tex,
                sampler: *gpu.Sampler,

                fn init(device: *gpu.Device, size: gpu.Extent3D, format: gpu.Texture.Format, comptime label: [:0]const u8) @This() {
                    return .{
                        .tex = Channel.Tex.init(device, size, format, label),
                        .sampler = device.createSampler(&.{}),
                    };
                }

                fn release(self: *@This()) void {
                    self.tex.release();
                    self.sampler.release();
                }
            };
            const TextureMap = struct {
                hm: std.StringHashMap(Channel.Tex),

                fn init(device: *gpu.Device, queue: *gpu.Queue, at: *Shadertoy.ActiveToy) @This() {
                    const hm = std.StringHashMap(Channel.Tex).init(allocator);
                    var self = @This(){
                        .hm = hm,
                    };
                    self.insert_textures(device, queue, &at.passes.image.inputs);
                    if (at.passes.buffer1) |*buf| self.insert_textures(device, queue, &buf.inputs);
                    if (at.passes.buffer2) |*buf| self.insert_textures(device, queue, &buf.inputs);
                    if (at.passes.buffer3) |*buf| self.insert_textures(device, queue, &buf.inputs);
                    if (at.passes.buffer4) |*buf| self.insert_textures(device, queue, &buf.inputs);
                    return self;
                }

                fn insert_textures(self: *@This(), device: *gpu.Device, queue: *gpu.Queue, inputs: *Shadertoy.ActiveToy.Inputs) void {
                    if (inputs.input1) |*inp| self.insert_texture(device, queue, inp);
                    if (inputs.input2) |*inp| self.insert_texture(device, queue, inp);
                    if (inputs.input3) |*inp| self.insert_texture(device, queue, inp);
                    if (inputs.input4) |*inp| self.insert_texture(device, queue, inp);
                }

                fn insert_texture(self: *@This(), device: *gpu.Device, queue: *gpu.Queue, input: *Shadertoy.ActiveToy.Input) void {
                    switch (input.typ) {
                        .texture => |*tex| {
                            // MEH: don't really want to deal with allocation falures here
                            // better solution would be using a big enough arena allocator
                            const res = self.hm.getOrPut(tex.name) catch unreachable;
                            if (!res.found_existing) {
                                res.key_ptr.* = allocator.dupe(u8, tex.name) catch unreachable;
                                res.value_ptr.* = Channel.Tex.from_img(device, queue, &tex.img, tex.name);
                            }
                        },
                        else => {},
                    }
                }

                fn release(self: *@This()) void {
                    var it = self.hm.iterator();
                    while (it.next()) |val| {
                        allocator.free(val.key_ptr.*);
                        val.value_ptr.*.release();
                    }
                    self.hm.deinit();
                }
            };
            bufferA: Channel,
            bufferB: Channel,
            bufferC: Channel,
            bufferD: Channel,
            keyboard: Channel.Tex,
            sound: Channel.Tex,
            textures: TextureMap,

            screen: Sampled,
            empty_input: Sampled,

            fn init(device: *gpu.Device, queue: *gpu.Queue, size: gpu.Extent3D, at: *Shadertoy.ActiveToy) @This() {
                return .{
                    .bufferA = Channel.init(device, size, .rgba32_float, "buffer A"),
                    .bufferB = Channel.init(device, size, .rgba32_float, "buffer B"),
                    .bufferC = Channel.init(device, size, .rgba32_float, "buffer C"),
                    .bufferD = Channel.init(device, size, .rgba32_float, "buffer D"),
                    .keyboard = Channel.Tex.init(device, size, .rgba32_float, "keyboard buffer"),
                    .sound = Channel.Tex.init(device, size, .rgba32_float, "sound buffer"),
                    .screen = Sampled.init(device, size, .rgba32_float, "screen buffer"),
                    .empty_input = Sampled.init(device, .{ .width = 1 }, .rgba32_float, "empty buffer"),
                    .textures = TextureMap.init(device, queue, at),
                };
            }

            fn release(self: *@This()) void {
                self.bufferA.release();
                self.bufferB.release();
                self.bufferC.release();
                self.bufferD.release();
                self.keyboard.release();
                self.sound.release();
                self.screen.release();
                self.empty_input.release();
                self.textures.release();
            }

            fn get_current(self: *@This(), buf: Shadertoy.ActiveToy.Buffer) *Channel.Tex {
                switch (buf) {
                    .Buf => |buf1| switch (buf1) {
                        .BufferA => return &self.bufferA.current,
                        .BufferB => return &self.bufferB.current,
                        .BufferC => return &self.bufferC.current,
                        .BufferD => return &self.bufferD.current,
                    },
                    .keyboard => return &self.keyboard,
                    .music => return &self.sound,
                    .texture => |tex| return self.textures.hm.getPtr(tex.name).?,
                }
            }

            fn get_last_frame(self: *@This(), buf: Shadertoy.ActiveToy.Buffer) *Channel.Tex {
                switch (buf) {
                    .Buf => |buf1| switch (buf1) {
                        .BufferA => return &self.bufferA.last_frame,
                        .BufferB => return &self.bufferB.last_frame,
                        .BufferC => return &self.bufferC.last_frame,
                        .BufferD => return &self.bufferD.last_frame,
                    },
                    .keyboard => return &self.keyboard,
                    .music => return &self.sound,
                    .texture => |tex| return self.textures.hm.getPtr(tex.name).?,
                }
            }

            fn current_view(self: *@This(), buf: Shadertoy.ActiveToy.Buf) *gpu.TextureView {
                return self.get_current(.{ .Buf = buf }).view;
            }
        };

        pass: struct {
            screen: ScreenPass,
            image: ImagePass,
            buffer1: ?Pass = null,
            buffer2: ?Pass = null,
            buffer3: ?Pass = null,
            buffer4: ?Pass = null,

            fn init() !@This() {
                return .{};
            }

            fn release(self: *@This()) void {
                self.screen.release();
                self.image.release();
                if (self.buffer1) |*buf| buf.release();
                if (self.buffer2) |*buf| buf.release();
                if (self.buffer3) |*buf| buf.release();
                if (self.buffer4) |*buf| buf.release();
            }
        },
        buffers: Buffers,
        binding: Binding,
        uniforms: ShadertoyUniformBuffers,

        fn init(at: *Shadertoy.ActiveToy, core_mod: *mach.Core.Mod, size: gpu.Extent3D, compiler: *Shadertoy.ToyMan.Compiler) !@This() {
            const core: *mach.Core = core_mod.state();
            const device: *gpu.Device = core.device;

            const uniforms = try ShadertoyUniformBuffers.init(device, size.height, size.width);

            var binding = Binding{ .label = "fajdl;f;jda" };
            errdefer binding.deinit(allocator);
            // LEAK: if .add() errors, those uniforms don't get deinited
            try binding.add(uniforms.uniforms, allocator);
            // TODO:
            // try binding.init(@tagName(name), device, allocator);

            var buffers = Buffers.init(device, core.queue, size, at);
            errdefer buffers.release();

            const vert = try compiler.ctx.compile_vert();
            defer allocator.free(vert);
            var shader = try compiler.ctx.compile(.screen, at.has_common);
            defer allocator.free(shader);
            var screen_pass = try ScreenPass.init(core_mod, &binding, buffers.screen.tex.view, buffers.screen.sampler, &.{ .vert = vert, .frag = shader });
            errdefer screen_pass.release();

            allocator.free(shader);
            shader = try compiler.ctx.compile(.image, at.has_common);
            var image_pass = try ImagePass.init(at, core_mod, &buffers, &binding, &.{ .vert = vert, .frag = shader });
            errdefer image_pass.release();

            var pass1 = if (at.passes.buffer1) |*pass| blk: {
                allocator.free(shader);
                shader = try compiler.ctx.compile(.buffer1, at.has_common);
                break :blk try Pass.init(core_mod, pass, &buffers, &binding, &.{ .vert = vert, .frag = shader });
            } else null;
            errdefer if (pass1) |*pass| pass.release();

            var pass2 = if (at.passes.buffer2) |*pass| blk: {
                allocator.free(shader);
                shader = try compiler.ctx.compile(.buffer2, at.has_common);
                break :blk try Pass.init(core_mod, pass, &buffers, &binding, &.{ .vert = vert, .frag = shader });
            } else null;
            errdefer if (pass2) |*pass| pass.release();

            var pass3 = if (at.passes.buffer3) |*pass| blk: {
                allocator.free(shader);
                shader = try compiler.ctx.compile(.buffer3, at.has_common);
                break :blk try Pass.init(core_mod, pass, &buffers, &binding, &.{ .vert = vert, .frag = shader });
            } else null;
            errdefer if (pass3) |*pass| pass.release();

            var pass4 = if (at.passes.buffer4) |*pass| blk: {
                allocator.free(shader);
                shader = try compiler.ctx.compile(.buffer4, at.has_common);
                break :blk try Pass.init(core_mod, pass, &buffers, &binding, &.{ .vert = vert, .frag = shader });
            } else null;
            errdefer if (pass4) |*pass| pass.release();

            return .{
                .pass = .{
                    .screen = screen_pass,
                    .image = image_pass,
                    .buffer1 = pass1,
                    .buffer2 = pass2,
                    .buffer3 = pass3,
                    .buffer4 = pass4,
                },
                .uniforms = uniforms,
                .binding = binding,
                .buffers = buffers,
            };
        }

        fn update(self: *@This(), core_mod: *mach.Core.Mod, comp: *Shadertoy.ToyMan.Compiled) !void {
            const core: *mach.Core = core_mod.state();
            const device = core.device;

            comp.mutex.lock();
            defer comp.mutex.unlock();

            if (comp.input_fuse.unfuse()) {
                self.resize(device, core.queue, &comp.toy);
            }

            if (comp.uniform_reset_fuse.unfuse()) {
                self.uniforms.uniforms.val.reset_state();
            }

            var new: @TypeOf(self.pass) = blk: {
                var image = try ImagePass.init(&comp.toy, core_mod, &self.buffers, &self.binding, &.{ .vert = comp.vert, .frag = comp.image });
                errdefer image.release();
                var screen = try ScreenPass.init(core_mod, &self.binding, self.buffers.screen.tex.view, self.buffers.screen.sampler, &.{ .vert = comp.vert, .frag = comp.screen });
                errdefer screen.release();

                break :blk .{
                    .image = image,
                    .screen = screen,
                };
            };
            errdefer new.release();

            if (comp.toy.passes.buffer1) |*buf| {
                new.buffer1 = try Pass.init(core_mod, buf, &self.buffers, &self.binding, &.{ .vert = comp.vert, .frag = comp.buffer1.? });
            }
            if (comp.toy.passes.buffer2) |*buf| {
                new.buffer2 = try Pass.init(core_mod, buf, &self.buffers, &self.binding, &.{ .vert = comp.vert, .frag = comp.buffer2.? });
            }
            if (comp.toy.passes.buffer3) |*buf| {
                new.buffer3 = try Pass.init(core_mod, buf, &self.buffers, &self.binding, &.{ .vert = comp.vert, .frag = comp.buffer3.? });
            }
            if (comp.toy.passes.buffer4) |*buf| {
                new.buffer4 = try Pass.init(core_mod, buf, &self.buffers, &self.binding, &.{ .vert = comp.vert, .frag = comp.buffer4.? });
            }

            self.pass.release();
            self.pass = new;
        }

        fn resize(self: *@This(), device: *gpu.Device, queue: *gpu.Queue, at: *Shadertoy.ActiveToy) void {
            self.buffers.release();
            self.buffers = Buffers.init(device, queue, .{
                .width = self.uniforms.uniforms.val.width,
                .height = self.uniforms.uniforms.val.height,
            }, at);
        }

        fn swap(self: *@This()) void {
            self.buffers.bufferA.swap();
            self.buffers.bufferB.swap();
            self.buffers.bufferC.swap();
            self.buffers.bufferD.swap();
            if (self.pass.buffer1) |*buf| buf.swap();
            if (self.pass.buffer2) |*buf| buf.swap();
            if (self.pass.buffer3) |*buf| buf.swap();
            if (self.pass.buffer4) |*buf| buf.swap();
        }

        fn render(self: *@This(), encoder: *gpu.CommandEncoder, screen_channel: *Channel.Tex, screen: *gpu.TextureView) void {
            if (self.pass.buffer1) |*buf| buf.render(encoder, self.buffers.current_view(buf.output));
            if (self.pass.buffer2) |*buf| buf.render(encoder, self.buffers.current_view(buf.output));
            if (self.pass.buffer3) |*buf| buf.render(encoder, self.buffers.current_view(buf.output));
            if (self.pass.buffer4) |*buf| buf.render(encoder, self.buffers.current_view(buf.output));
            self.pass.image.render(encoder, screen_channel.view);
            self.pass.screen.render(encoder, screen);
        }

        fn release(self: *@This()) void {
            self.pass.release();

            self.buffers.release();
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
        err_channel: utils.Channel(?[]const u8),

        fn init(alloc: std.mem.Allocator) !*@This() {
            const self = try alloc.create(@This());
            self.* = .{
                .ticks = std.atomic.Value(u32).init(0),
                .err_fuse = .{},
                .err_channel = try utils.Channel(?[]const u8).init(alloc),
            };
            return self;
        }

        fn tick(self: *@This()) !void {
            _ = self.ticks.fetchAdd(1, .acq_rel);
            const ticks = self.ticks.fetchMin(tick_threshold, .acq_rel);
            if (ticks >= tick_threshold) {
                const was_fused = self.err_fuse.unfuse();

                if (was_fused) {
                    try self.err_channel.send(null);
                }
            }
        }

        fn fuse(self: *@This()) bool {
            self.ticks.store(0, .release);
            return self.err_fuse.fuse();
        }

        fn check(self: *@This()) bool {
            return self.err_fuse.check();
        }

        fn deinit(self: *@This()) void {
            const alloc = self.err_channel.pinned.dq.allocator;
            while (self.err_channel.try_recv()) |msg| {
                if (msg) |nonull| {
                    alloc.free(nonull);
                }
            }
            self.err_channel.deinit();
            alloc.destroy(self);
        }
    };

    pub const Config = struct {
        shader_compile_opt: compile.Glslc.Compiler.Opt = .fast,
        shader_dump_assembly: bool = false,
        pause_shader: bool = false,
        ignore_pause_fuse: Fuse = .{},
        ignore_pause_fuse_swapped: Fuse = .{},
        resize_fuse: Fuse = .{},
        gpu_err_fuse: *GpuFuse,
    };

    timer: mach.Timer,
    toyman: Shadertoy.ToyMan,
    pri: PlaygroundRenderInfo,
    config: *Config,

    fn deinit(self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        self.toyman.deinit();
        self.pri.release();
        self.config.gpu_err_fuse.deinit();
        allocator.destroy(self.config);
    }

    fn init(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const core: *mach.Core = core_mod.state();
        const device = core.device;

        const render_config = try allocator.create(Config);
        errdefer allocator.destroy(render_config);
        render_config.* = .{
            .gpu_err_fuse = try GpuFuse.init(allocator),
        };

        var man = try Shadertoy.ToyMan.init(render_config);
        errdefer man.deinit();

        const height = core_mod.get(core.main_window, .height).?;
        const width = core_mod.get(core.main_window, .width).?;
        const size = gpu.Extent3D{
            .width = width,
            .height = height,
            .depth_or_array_layers = 1,
        };

        // TODO: this initialzation should be all from @embed() shaders
        try man.load_zhadertoy("new");
        var pri = try PlaygroundRenderInfo.init(man.active_toy, core_mod, size, &man.compiler);
        errdefer pri.release();

        // - [machengine.org/content/engine/gpu/errors.md](https://github.com/hexops/machengine.org/blob/e10e4245c4c1de7fe9e4882e378a86d42eb1035a/content/engine/gpu/errors.md)
        device.setUncapturedErrorCallback(render_config.gpu_err_fuse, struct {
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
                    const msg = std.fmt.allocPrint(ctx.err_channel.pinned.dq.allocator, "Gpu Validation Error: {s}\n", .{message}) catch |err| {
                        std.debug.print("Fatal Error: {any}\n", .{err});
                        std.process.exit(1);
                    };
                    ctx.err_channel.send(msg) catch |err| {
                        std.debug.print("Fatal Error: {any}\n", .{err});
                        std.process.exit(1);
                    };
                }
            }
        }.callback);

        self_mod.init(.{
            .timer = try mach.Timer.start(),
            .toyman = man,

            .pri = pri,
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

        self.pri.binding.update(encoder);

        var ignore = self.config.ignore_pause_fuse.unfuse();
        if (ignore) {
            _ = self.config.ignore_pause_fuse_swapped.fuse();
        } else {
            ignore = self.config.ignore_pause_fuse_swapped.unfuse();
        }
        if (!self.config.pause_shader or ignore) {
            self.pri.swap();
            self.pri.render(encoder, &self.pri.buffers.screen.tex, app.rendering_data.?.screen);
        } else {
            self.pri.pass.screen.render(encoder, app.rendering_data.?.screen);
        }

        var command = encoder.finish(&.{ .label = label });
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});

        if (self.config.resize_fuse.unfuse()) {
            _ = self.toyman.compiler.fuse_resize();
        }

        try self.config.gpu_err_fuse.tick();
    }

    fn update_shaders(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const self: *@This() = self_mod.state();

        if (self.toyman.compiler.unfuse()) |compiled| {
            self.pri.update(core_mod, compiled) catch |e| {
                std.debug.print("Error while updating shaders: {any}\n", .{e});
            };
            _ = self.config.gpu_err_fuse.err_fuse.unfuse();
            _ = self.config.ignore_pause_fuse.fuse();
        }
    }

    fn tick(self_mod: *Mod, app_mod: *App.Mod) !void {
        const app: *App = app_mod.state();

        const self: *@This() = self_mod.state();
        if (self.toyman.has_updates()) {
            self_mod.schedule(.update_shaders);
        }

        var state = &self.pri.uniforms.uniforms.val;
        state.time_delta = self.timer.lap();
        if (!self.config.pause_shader) {
            state.time += state.time_delta;
        }
        state.frame += 1;

        for (app.events.items) |event| {
            switch (event) {
                .mouse_motion => |mouse| {
                    if (state.mouse_left) {
                        state.mouse_x = @floatCast(mouse.pos.x);
                        state.mouse_y = @floatCast(@as(f64, @floatFromInt(state.height)) - mouse.pos.y);
                        state.mouse.x = state.mouse_x;
                        state.mouse.y = state.mouse_y;
                        _ = self.config.ignore_pause_fuse.fuse();
                    }
                },
                .mouse_press => |button| {
                    switch (button.button) {
                        .left => {
                            state.mouse_left = true;
                            state.mouse.z = 1.0;
                        },
                        .right => {
                            state.mouse_right = true;
                            state.mouse.w = 1.0;
                        },
                        .middle => {
                            state.mouse_middle = true;
                        },
                        else => {},
                    }
                    state.mouse_x = @floatCast(button.pos.x);
                    state.mouse_y = @floatCast(@as(f64, @floatFromInt(state.height)) - button.pos.y);
                    state.mouse.x = state.mouse_x;
                    state.mouse.y = state.mouse_y;
                    _ = self.config.ignore_pause_fuse.fuse();
                },
                .mouse_release => |button| {
                    switch (button.button) {
                        .left => {
                            state.mouse_left = false;
                            state.mouse.z = 0.0;
                        },
                        .right => {
                            state.mouse_right = false;
                            state.mouse.w = 0.0;
                        },
                        .middle => {
                            state.mouse_middle = false;
                        },
                        else => {},
                    }
                    state.mouse_x = @floatCast(button.pos.x);
                    state.mouse_y = @floatCast(@as(f64, @floatFromInt(state.height)) - button.pos.y);
                    state.mouse.x = state.mouse_x;
                    state.mouse.y = state.mouse_y;
                },
                .mouse_scroll => |del| {
                    _ = del;
                    // state.scroll += del.yoffset;
                },
                .key_press => |ev| {
                    switch (ev.key) {
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
                    state.resolution.x = @floatFromInt(sze.width);
                    state.resolution.y = @floatFromInt(sze.height);
                    _ = self.config.resize_fuse.fuse();
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
        // NOTE: this code is core_mod.init
        {
            const core: *mach.Core = core_mod.state();
            try mach.core.init(.{
                .display_mode = if (core_mod.get(core.main_window, .fullscreen).?) .fullscreen else .windowed,
                .size = .{
                    .width = core_mod.get(core.main_window, .width).?,
                    .height = core_mod.get(core.main_window, .height).?,
                },
                .power_preference = .high_performance,
                .required_features = &[_]gpu.FeatureName{
                    // - [dawn float32 filtering](https://developer.chrome.com/blog/new-in-webgpu-119)
                    .float32_filterable,
                },
            });

            try core_mod.set(core.main_window, .framebuffer_format, mach.core.descriptor.format);
            try core_mod.set(core.main_window, .framebuffer_width, mach.core.descriptor.width);
            try core_mod.set(core.main_window, .framebuffer_height, mach.core.descriptor.height);
            try core_mod.set(core.main_window, .width, mach.core.size().width);
            try core_mod.set(core.main_window, .height, mach.core.size().height);

            core.allocator = mach.core.allocator;
            core.device = mach.core.device;
            core.queue = mach.core.device.getQueue();
        }
        // core_mod.schedule(.init);
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
                .key_press => |ev| {
                    switch (ev.key) {
                        .escape => {
                            core_mod.schedule(.exit);
                        },
                        else => {},
                    }
                },
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

        const color = utils.ColorParse.hex_xyzw;
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
        style.colors[imgui.Col_Header] = color(imgui.Vec4, "#3c3836ff");
        style.colors[imgui.Col_HeaderHovered] = color(imgui.Vec4, "#504945ff");
        style.colors[imgui.Col_HeaderActive] = color(imgui.Vec4, "#7c6f64ff");
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

    var error_msg_buf = std.mem.zeroes([1024:0]u8);
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
                if (imgui.collapsingHeader("shader compilation", imgui.TreeNodeFlags_None)) {
                    _ = imgui.checkbox("shader dump assembly", &renderer.config.shader_dump_assembly);

                    enum_checkbox(&renderer.config.shader_compile_opt, "shader compile opt mode");
                }

                if (imgui.collapsingHeader("toys", imgui.TreeNodeFlags_DefaultOpen)) {
                    _ = imgui.checkbox("pause shader", &renderer.config.pause_shader);
                    pick_toy(renderer) catch |e| std.debug.print("{any}\n", .{e});
                }

                if (imgui.collapsingHeader("uniforms", imgui.TreeNodeFlags_DefaultOpen)) {
                    if (imgui.button("reset uniforms")) {
                        renderer.pri.uniforms.uniforms.val.reset_state();
                    }
                    const val = renderer.pri.uniforms.uniforms.val;
                    imgui.text("frame: %d time: %.1fs delta: %.3fs", val.frame, val.time, val.time_delta);
                    imgui.text("mouse_x: %.3f mouse_y: %.3f", val.mouse_x, val.mouse_y);
                    imgui.text("width: %d height: %d", val.width, val.height);
                }
            }

            while (renderer.config.gpu_err_fuse.err_channel.try_recv()) |err| {
                if (err) |msg| {
                    defer renderer.config.gpu_err_fuse.err_channel.pinned.dq.allocator.free(msg);
                    std.debug.print("{s}", .{msg});
                    std.mem.copyForwards(u8, &error_msg_buf, msg[0..@min(msg.len, error_msg_buf.len - 1)]);
                    error_msg_buf[@min(msg.len, error_msg_buf.len - 1)] = 0;
                } else {
                    error_msg_buf[0] = 0;
                }
            }
            if (imgui.collapsingHeader("errors", imgui.TreeNodeFlags_DefaultOpen)) {
                imgui.textWrapped("%s", &error_msg_buf);
            }

            // imgui.showDemoWindow(null);
        }

        const color_attachment = gpu.RenderPassColorAttachment{
            .view = app.rendering_data.?.screen,
            .clear_value = utils.ColorParse.hex_rgba(gpu.Color, "#282828ff"),
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

    var input_buf = std.mem.zeroes([256:0]u8);
    var pick_toy_index: c_int = 0;
    fn pick_toy(renderer: *Renderer) !void {
        if (imgui.button("reset shader")) {
            try renderer.toyman.reset_toy();
        }

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
            "s:3tsyzl: iq geoids",
            "s:7slfWX: uninitialized vars",
            "s:WdyGzy: idk. broken",
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

        imgui.setNextItemWidth(200);
        const enter = imgui.inputText("shadertoy id", &input_buf, input_buf.len, imgui.InputTextFlags_CharsNoBlank | imgui.InputTextFlags_EnterReturnsTrue);
        imgui.sameLine();
        if (enter or imgui.button("load")) {
            try renderer.toyman.load_shadertoy(std.mem.span(@as([*:0]u8, @ptrCast(&input_buf))));
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
        imgui.setNextItemWidth(200);
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
