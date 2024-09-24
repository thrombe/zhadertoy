const std = @import("std");

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
var allocator = gpa.allocator();

const Renderer = struct {
    pub const name = .renderer;
    pub const systems = .{
        .init = .{ .handler = init },
        .deinit = .{ .handler = deinit },
        .tick = .{ .handler = tick },
        .render_frame = .{ .handler = render_frame },
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

    const IBuffer = struct {
        ptr: *anyopaque,
        vtable: *const struct {
            update: *const fn (any_self: *anyopaque, encoder: *gpu.CommandEncoder) void,
            deinit: *const fn (any_self: *anyopaque, alloc: std.mem.Allocator) void,
            size: *const fn () usize,
            buf: *const fn (any_self: *anyopaque) *gpu.Buffer,
        },

        fn size(self: *@This()) usize {
            return self.vtable.size();
        }
        fn buf(self: *@This()) *gpu.Buffer {
            return self.vtable.buf(self.ptr);
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
            fn size() usize {
                return @sizeOf(typ);
            }
            fn buf(self: *@This()) *gpu.Buffer {
                return self.buffer;
            }
            fn update(self: *@This(), encoder: *gpu.CommandEncoder) void {
                encoder.writeBuffer(self.buffer, 0, &[_]typ{self.val});
            }
            fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
                self.buffer.release();
                alloc.destroy(self);
            }
            fn interface(self: *@This()) IBuffer {
                return .{
                    .ptr = self,
                    .vtable = &.{
                        .update = @ptrCast(&update),
                        .deinit = @ptrCast(&@This().deinit),
                        .size = @ptrCast(&size),
                        .buf = @ptrCast(&buf),
                    },
                };
            }
        };
    }
    const Binding = struct {
        label: [*:0]const u8,
        group: ?*gpu.BindGroup = null,
        layout: ?*gpu.BindGroupLayout = null,
        buffers: std.ArrayListUnmanaged(IBuffer) = .{},

        fn add(self: *@This(), buffer: anytype, alloc: std.mem.Allocator) !void {
            const itf: IBuffer = buffer.interface();
            try self.buffers.append(alloc, itf);
        }
        fn init(self: *@This(), label: [*:0]const u8, device: *gpu.Device, alloc: std.mem.Allocator) !void {
            var layout_entries = std.ArrayListUnmanaged(gpu.BindGroupLayout.Entry){};
            defer layout_entries.deinit(alloc);
            var bind_group_entries = std.ArrayListUnmanaged(gpu.BindGroup.Entry){};
            defer bind_group_entries.deinit(alloc);

            for (self.buffers.items, 0..) |*buf, i| {
                const bind_group_layout_entry = gpu.BindGroupLayout.Entry.buffer(
                    @intCast(i),
                    .{
                        .vertex = true,
                        .fragment = true,
                        .compute = true,
                    },
                    .uniform,
                    true,
                    0,
                );
                try layout_entries.append(alloc, bind_group_layout_entry);

                const bind_group_entry = gpu.BindGroup.Entry.buffer(@intCast(i), buf.buf(), 0, buf.size());
                try bind_group_entries.append(alloc, bind_group_entry);
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
            if (self.layout) |layout| {
                layout.release();
            }
            if (self.group) |group| {
                group.release();
            }
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
    };
    const ShadertoyUniforms = struct {
        time: *Buffer(f32),
        resolution: *Buffer(Vec3),
    };

    timer: mach.Timer,

    pipeline: *gpu.RenderPipeline,
    binding: Binding,
    buffer: *Buffer(StateUniform),
    st_buffers: ShadertoyUniforms,

    fn deinit(self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        self.pipeline.release();
        self.binding.deinit(allocator);

        // buffer's interface is already in self.binding
        // self.buffer.deinit(allocator);
        // self.st_buffers.time.deinit(allocator);
        // self.st_buffers.resolution.deinit(allocator);
    }

    fn init(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const core: *mach.Core = core_mod.state();
        const device = core.device;

        const state = .{
            .render_width = core_mod.get(core.main_window, .width).?,
            .render_height = core_mod.get(core.main_window, .height).?,
            .display_width = core_mod.get(core.main_window, .height).?,
            .display_height = core_mod.get(core.main_window, .height).?,
        };
        const buffer = try Buffer(StateUniform).new(@tagName(name), state, allocator, device);
        const st_buffers = ShadertoyUniforms{
            .time = try Buffer(f32).new(@tagName(name), 0.0, allocator, device),
            .resolution = try Buffer(Vec3).new(@tagName(name), Vec3{
                .x = @floatFromInt(core_mod.get(core.main_window, .width).?),
                .y = @floatFromInt(core_mod.get(core.main_window, .height).?),
                .z = 0.0,
            }, allocator, device),
        };

        var binding = Binding{ .label = buffer.label };
        try binding.add(buffer, allocator);
        try binding.add(st_buffers.time, allocator);
        try binding.add(st_buffers.resolution, allocator);
        try binding.init(@tagName(name), device, allocator);

        const bind_group_layouts = [_]*gpu.BindGroupLayout{binding.layout.?};
        const pipeline_layout = device.createPipelineLayout(&gpu.PipelineLayout.Descriptor.init(.{
            .label = binding.label,
            .bind_group_layouts = &bind_group_layouts,
        }));
        defer pipeline_layout.release();

        // const shader_module = device.createShaderModuleWGSL("shader.wgsl", @embedFile("shader.wgsl"));
        // defer shader_module.release();
        // const frag_shader_module = shader_module;
        // const vertex_shader_module = shader_module;

        const vert = @embedFile("vert.glsl");
        var compiler = Glslc.Compiler{ .opt = .fast };
        compiler.stage = .vertex;
        try compiler.dump_assembly(allocator, vert);
        const vertex_bytes = try compiler.compile(allocator, vert, .spirv);
        defer allocator.free(vertex_bytes);
        const vertex_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
            .next_in_chain = .{ .spirv_descriptor = &.{
                .code_size = @intCast(vertex_bytes.len),
                .code = vertex_bytes.ptr,
            } },
            .label = "vert.glsl",
        });
        defer vertex_shader_module.release();

        const frag = @embedFile("frag.glsl");
        compiler.stage = .fragment;
        try compiler.dump_assembly(allocator, frag);
        const frag_bytes = try compiler.compile(allocator, frag, .spirv);
        defer allocator.free(frag_bytes);
        const frag_shader_module = device.createShaderModule(&gpu.ShaderModule.Descriptor{
            .next_in_chain = .{ .spirv_descriptor = &.{
                .code_size = @intCast(frag_bytes.len),
                .code = frag_bytes.ptr,
            } },
            .label = "frag.glsl",
        });
        defer frag_shader_module.release();

        const blend = gpu.BlendState{};
        const color_target = gpu.ColorTargetState{
            .format = core_mod.get(core.main_window, .framebuffer_format).?,
            .blend = &blend,
            .write_mask = gpu.ColorWriteMaskFlags.all,
        };
        const fragment = gpu.FragmentState.init(.{
            .module = frag_shader_module,
            // .entry_point = "frag_main",
            .entry_point = "main",
            .targets = &.{color_target},
        });
        const vertex = gpu.VertexState{
            .module = vertex_shader_module,
            // .entry_point = "vert_main",
            .entry_point = "main",
        };

        const pipeline = device.createRenderPipeline(&gpu.RenderPipeline.Descriptor{
            .label = binding.label,
            .fragment = &fragment,
            .layout = pipeline_layout,
            .vertex = vertex,
        });

        self_mod.init(.{
            .timer = try mach.Timer.start(),

            .binding = binding,
            .buffer = buffer,
            .st_buffers = st_buffers,
            .pipeline = pipeline,
        });
    }

    fn render_frame(
        self_mod: *Mod,
        core_mod: *mach.Core.Mod,
    ) !void {
        const self: *@This() = self_mod.state();
        const core: *mach.Core = core_mod.state();
        const device: *gpu.Device = core.device;

        const back_buffer_view = mach.core.swap_chain.getCurrentTextureView().?;
        defer back_buffer_view.release();

        const label = @tagName(name) ++ ".render_frame";
        const encoder = device.createCommandEncoder(&.{ .label = label });
        defer encoder.release();

        // encoder.writeBuffer(self.uniform_buffer, 0, &[_]StateUniform{self.state});
        self.binding.update(encoder);

        const sky_blue_background = gpu.Color{ .r = 40.0 / 255.0, .g = 40.0 / 255.0, .b = 40.0 / 255.0, .a = 1 };
        const color_attachments = [_]gpu.RenderPassColorAttachment{.{
            .view = back_buffer_view,
            .clear_value = sky_blue_background,
            .load_op = .clear,
            .store_op = .store,
        }};
        const render_pass = encoder.beginRenderPass(&gpu.RenderPassDescriptor.init(.{
            .label = label,
            .color_attachments = &color_attachments,
        }));
        defer render_pass.release();

        render_pass.setPipeline(self.pipeline);
        render_pass.setBindGroup(0, self.binding.group.?, &.{ 0, 0, 0 });
        render_pass.draw(6, 1, 0, 0);
        render_pass.end();

        var command = encoder.finish(&.{ .label = label });
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});

        core_mod.schedule(.present_frame);
    }

    fn tick(self_mod: *Mod, core_mod: *mach.Core.Mod) !void {
        defer self_mod.schedule(.render_frame);

        const self: *@This() = self_mod.state();

        var state = &self.buffer.val;
        state.time += self.timer.lap();
        self.st_buffers.time.val = state.time;

        var iter = mach.core.pollEvents();
        while (iter.next()) |event| {
            switch (event) {
                .mouse_motion => |pos| {
                    state.cursor_x = @floatCast(pos.pos.x);
                    state.cursor_y = @floatCast(pos.pos.y);
                },
                .mouse_press => |button| {
                    switch (button.button) {
                        .left => {
                            state.mouse_left = 1;
                        },
                        .right => {
                            state.mouse_right = 1;
                        },
                        .middle => {
                            state.mouse_middle = 1;
                        },
                        else => {},
                    }
                },
                .mouse_release => |button| {
                    switch (button.button) {
                        .left => {
                            state.mouse_left = 0;
                        },
                        .right => {
                            state.mouse_right = 0;
                        },
                        .middle => {
                            state.mouse_middle = 0;
                        },
                        else => {},
                    }
                },
                .mouse_scroll => |del| {
                    state.scroll += del.yoffset;
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
                    state.render_height = sze.height;
                    state.render_width = sze.width;
                    state.display_height = sze.height;
                    state.display_width = sze.width;
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

// TODO: move this to a mach "entrypoint" zig module
pub fn main() !void {
    // try Glslang.testfn();
    // try Dawn.testfn();
    // try Shaderc.testfn();
    // try Glslc.testfn();

    // Initialize mach core
    try mach.core.initModule();

    // Main loop
    while (try mach.core.tick()) {}
}

const Glslc = struct {
    fn testfn() !void {
        const compiler = Compiler{
            .opt = .fast,
            .stage = .fragment,
        };
        const shader: []const u8 = @embedFile("frag.glsl");
        const bytes = try compiler.compile(allocator, shader, .assembly);
        defer allocator.free(bytes);

        std.debug.print("{s}\n", .{bytes});
    }

    const Compiler = struct {
        opt: enum {
            none,
            small,
            fast,
        } = .none,
        lang: enum {
            glsl,
            hlsl,
        } = .glsl,
        stage: enum {
            vertex,
            fragment,
            compute,
        } = .fragment,
        const OutputType = enum {
            assembly,
            spirv,
        };

        fn dump_assembly(self: @This(), alloc: std.mem.Allocator, code: []const u8) !void {
            const bytes = try self.compile(alloc, code, .assembly);
            defer alloc.free(bytes);
            std.debug.print("{s}\n", .{bytes});
        }

        fn compile(self: @This(), alloc: std.mem.Allocator, code: []const u8, comptime output_type: OutputType) !(switch (output_type) {
            .spirv => []u32,
            .assembly => []u8,
        }) {
            var args = std.ArrayList([]const u8).init(alloc);
            defer {
                for (args.items) |arg| {
                    alloc.free(arg);
                }
                args.deinit();
            }
            try args.append(try alloc.dupe(u8, @as([]const u8, "glslc")));
            try args.append(try alloc.dupe(u8, try std.fmt.allocPrint(alloc, "-fshader-stage={s}", .{switch (self.stage) {
                .fragment => "fragment"[0..],
                .vertex => "vertex"[0..],
                .compute => "compute"[0..],
            }})));
            try args.append(try alloc.dupe(u8, try std.fmt.allocPrint(alloc, "-x{s}", .{switch (self.lang) {
                .glsl => @as([]const u8, "glsl"),
                .hlsl => @as([]const u8, "hlsl"),
            }})));
            if (output_type == .assembly) {
                try args.append(try alloc.dupe(u8, @as([]const u8, "-S")));
            }
            try args.append(try alloc.dupe(u8, try std.fmt.allocPrint(alloc, "-O{s}", .{switch (self.opt) {
                .fast => @as([]const u8, ""),
                .small => @as([]const u8, "s"),
                .none => @as([]const u8, "0"),
            }})));
            try args.append(try alloc.dupe(u8, @as([]const u8, "-o-")));
            try args.append(try alloc.dupe(u8, @as([]const u8, "-")));

            var child = std.process.Child.init(args.items, alloc);
            child.stdin_behavior = .Pipe;
            child.stdout_behavior = .Pipe;
            child.stderr_behavior = .Pipe;

            try child.spawn();

            const stdin = child.stdin orelse return error.NoStdin;
            child.stdin = null;
            const stdout = child.stdout orelse return error.NoStdout;
            child.stdout = null;
            const stderr = child.stderr orelse return error.NoStderr;
            child.stderr = null;
            defer stdout.close();
            defer stderr.close();

            try stdin.writeAll(code);
            stdin.close();

            const err = try child.wait();
            errdefer {
                const msg = stderr.readToEndAlloc(alloc, 10 * 1000) catch unreachable;
                std.debug.print("{s}\n", .{msg});
                alloc.free(msg);
            }
            switch (err) {
                .Exited => |e| {
                    if (e != 0) {
                        std.debug.print("exited with code: {}\n", .{e});
                        return error.GlslcErroredOut;
                    }
                },
                // .Signal => |code| {},
                // .Stopped => |code| {},
                // .Unknown => |code| {},
                else => |e| {
                    std.debug.print("exited with code: {}\n", .{e});
                    return error.GlslcErroredOut;
                },
            }

            const bytes = try stdout.readToEndAllocOptions(
                alloc,
                100 * 1000,
                null,
                4,
                null,
            );
            return switch (output_type) {
                .spirv => std.mem.bytesAsSlice(u32, bytes),
                .assembly => bytes,
            };
        }
    };
};

const Shaderc = struct {
    const C = @cImport({
        @cInclude("shaderc/shaderc.h");
    });

    const Shader = struct {
        result: *C.struct_shaderc_compilation_result,

        fn deinit(self: *@This()) void {
            C.shaderc_result_release(self.result);
        }

        /// owned by Shader
        fn bytes(self: *@This()) []const u32 {
            const len = C.shaderc_result_get_length(self.result);
            const res = C.shaderc_result_get_bytes(self.result)[0..len];
            // alignment guranteed when it is spirv
            return @ptrCast(@alignCast(res));
        }
    };

    fn testfn() !void {
        const shader = @embedFile("frag.glsl");
        // std.debug.print("{s}\n", .{shader.ptr[0..shader.len]});

        const compiler = C.shaderc_compiler_initialize() orelse return error.CouldNotInitShaderc;
        defer C.shaderc_compiler_release(compiler);

        const options = C.shaderc_compile_options_initialize() orelse return error.CouldNotInitShaderc;
        defer C.shaderc_compile_options_release(options);

        C.shaderc_compile_options_set_optimization_level(options, C.shaderc_optimization_level_performance);
        C.shaderc_compile_options_set_source_language(options, C.shaderc_source_language_glsl);
        // C.shaderc_compile_options_set_vulkan_rules_relaxed(options, true);

        const result = C.shaderc_compile_into_spv_assembly(
            compiler,
            shader.ptr,
            shader.len,
            C.shaderc_glsl_fragment_shader,
            "frag.glsl",
            "main",
            options,
        ) orelse return error.CouldNotCompileShader;
        errdefer C.shaderc_result_release(result);

        const status = C.shaderc_result_get_compilation_status(result);
        if (status != C.shaderc_compilation_status_success) {
            const msg = C.shaderc_result_get_error_message(result);
            const err = std.mem.span(msg);
            std.debug.print("{s}\n", .{err});
            return error.CouldNotCompileShader;
        }

        const len = C.shaderc_result_get_length(result);
        const bytes = C.shaderc_result_get_bytes(result)[0..len];
        std.debug.print("{s}\n", .{bytes});
    }
};

const Dawn = struct {
    const C = @cImport({
        @cInclude("dawn/webgpu.h");
    });

    fn testfn() !void {}
};

const Glslang = struct {
    const C = @cImport({
        @cInclude("glslang/Include/glslang_c_interface.h");
        @cInclude("glslang/Public/resource_limits_c.h");
        @cInclude("stdio.h");
        @cInclude("stdlib.h");
        @cInclude("stdint.h");
        @cInclude("stdbool.h");
    });

    const SpirVBinary = struct {
        words: ?*C.u_int32_t, // SPIR-V words
        size: usize, // number of words in SPIR-V binary
    };

    fn testfn() !void {
        const shader = @embedFile("shader.glsl");
        const bin = try compileShaderToSPIRV_Vulkan(
            C.GLSLANG_STAGE_VERTEX_MASK | C.GLSLANG_STAGE_FRAGMENT_MASK | C.GLSLANG_STAGE_COMPUTE_MASK,
            shader.ptr,
        );
        _ = bin;
    }

    fn compileShaderToSPIRV_Vulkan(stage: C.glslang_stage_t, shaderSource: [*c]const u8) !SpirVBinary {
        const input = C.glslang_input_t{
            .language = C.GLSLANG_SOURCE_GLSL,
            .stage = stage,
            .client = C.GLSLANG_CLIENT_VULKAN,
            .client_version = C.GLSLANG_TARGET_VULKAN_1_2,
            .target_language = C.GLSLANG_TARGET_SPV,
            .target_language_version = C.GLSLANG_TARGET_SPV_1_5,
            .code = shaderSource,
            .default_version = 100,
            .default_profile = C.GLSLANG_NO_PROFILE,
            .force_default_version_and_profile = C.false,
            .forward_compatible = C.false,
            .messages = C.GLSLANG_MSG_DEFAULT_BIT,
            .resource = C.glslang_default_resource(),
        };

        const shader = C.glslang_shader_create(&input);

        var bin = SpirVBinary{
            .words = null,
            .size = 0,
        };
        if (C.glslang_shader_preprocess(shader, &input) == C.false) {
            _ = C.printf("GLSL preprocessing failed\n");
            _ = C.printf("%s\n", C.glslang_shader_get_info_log(shader));
            _ = C.printf("%s\n", C.glslang_shader_get_info_debug_log(shader));
            _ = C.printf("%s\n", input.code);
            C.glslang_shader_delete(shader);
            return bin;
        }

        if (C.glslang_shader_parse(shader, &input) == C.false) {
            _ = C.printf("GLSL parsing failed\n");
            _ = C.printf("%s\n", C.glslang_shader_get_info_log(shader));
            _ = C.printf("%s\n", C.glslang_shader_get_info_debug_log(shader));
            _ = C.printf("%s\n", C.glslang_shader_get_preprocessed_code(shader));
            C.glslang_shader_delete(shader);
            return bin;
        }

        const program = C.glslang_program_create();
        C.glslang_program_add_shader(program, shader);

        if (C.glslang_program_link(program, C.GLSLANG_MSG_SPV_RULES_BIT | C.GLSLANG_MSG_VULKAN_RULES_BIT) == C.false) {
            _ = C.printf("GLSL linking failed\n");
            _ = C.printf("%s\n", C.glslang_program_get_info_log(program));
            _ = C.printf("%s\n", C.glslang_program_get_info_debug_log(program));
            C.glslang_program_delete(program);
            C.glslang_shader_delete(shader);
            return bin;
        }

        C.glslang_program_SPIRV_generate(program, stage);

        bin.size = C.glslang_program_SPIRV_get_size(program);
        bin.words = @ptrCast(@alignCast(C.malloc(bin.size * @sizeOf(C.u_int32_t))));
        C.glslang_program_SPIRV_get(program, bin.words);

        const spirv_messages = @as(?[*c]const u8, C.glslang_program_SPIRV_get_messages(program));
        if (spirv_messages) |msg| {
            _ = C.printf("%s\n", msg);
        }

        C.glslang_program_delete(program);
        C.glslang_shader_delete(shader);

        return bin;
    }
};
