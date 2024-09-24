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

    timer: mach.Timer,

    bind_group: *gpu.BindGroup,
    pipeline: *gpu.RenderPipeline,
    uniform_buffer: *gpu.Buffer,
    state: StateUniform,

    fn deinit(self_mod: *Mod) !void {
        const self: *@This() = self_mod.state();
        self.pipeline.release();
        self.bind_group.release();
        self.uniform_buffer.release();
    }

    fn init(
        core_mod: *mach.Core.Mod,
        self_mod: *Mod,
    ) !void {
        const core: *mach.Core = core_mod.state();
        const device = core.device;

        // const shader_module = device.createShaderModuleWGSL("shader.wgsl", @embedFile("shader.wgsl"));
        // defer shader_module.release();

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
            .entry_point = "main",
            .targets = &.{color_target},
        });
        const vertex = gpu.VertexState{
            .module = vertex_shader_module,
            .entry_point = "main",
        };

        const label = @tagName(name) ++ ".init";
        const uniform_buffer = device.createBuffer(&.{
            .label = label ++ " uniform buffer",
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(StateUniform) * uniform_offset, // HUH: why is this size*uniform_offset? it should be the next multiple of uniform_offset
            .mapped_at_creation = .false,
        });

        const bind_group_layout_entry = gpu.BindGroupLayout.Entry.buffer(
            0,
            .{
                .vertex = true,
                .fragment = true,
                .compute = true,
            },
            .uniform,
            true,
            0,
        );
        const bind_group_layout = device.createBindGroupLayout(
            &gpu.BindGroupLayout.Descriptor.init(.{
                .label = label,
                .entries = &.{bind_group_layout_entry},
            }),
        );
        defer bind_group_layout.release();

        const bind_group = device.createBindGroup(
            &gpu.BindGroup.Descriptor.init(.{
                .label = label,
                .layout = bind_group_layout,
                .entries = &.{
                    gpu.BindGroup.Entry.buffer(0, uniform_buffer, 0, @sizeOf(StateUniform)),
                },
            }),
        );

        const bind_group_layouts = [_]*gpu.BindGroupLayout{bind_group_layout};
        const pipeline_layout = device.createPipelineLayout(&gpu.PipelineLayout.Descriptor.init(.{
            .label = label,
            .bind_group_layouts = &bind_group_layouts,
        }));
        defer pipeline_layout.release();

        const pipeline = device.createRenderPipeline(&gpu.RenderPipeline.Descriptor{
            .label = label,
            .fragment = &fragment,
            .layout = pipeline_layout,
            .vertex = vertex,
        });

        self_mod.init(.{
            .timer = try mach.Timer.start(),

            .bind_group = bind_group,
            .pipeline = pipeline,
            .uniform_buffer = uniform_buffer,
            .state = .{
                .render_width = core_mod.get(core.main_window, .width).?,
                .render_height = core_mod.get(core.main_window, .height).?,
                .display_width = core_mod.get(core.main_window, .height).?,
                .display_height = core_mod.get(core.main_window, .height).?,
            },
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

        encoder.writeBuffer(self.uniform_buffer, 0, &[_]StateUniform{self.state});

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
        render_pass.setBindGroup(0, self.bind_group, &.{0});
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

        self.state.time += self.timer.lap();

        var iter = mach.core.pollEvents();
        while (iter.next()) |event| {
            switch (event) {
                .mouse_motion => |pos| {
                    self.state.cursor_x = @floatCast(pos.pos.x);
                    self.state.cursor_y = @floatCast(pos.pos.y);
                },
                .mouse_press => |button| {
                    switch (button.button) {
                        .left => {
                            self.state.mouse_left = 1;
                        },
                        .right => {
                            self.state.mouse_right = 1;
                        },
                        .middle => {
                            self.state.mouse_middle = 1;
                        },
                        else => {},
                    }
                },
                .mouse_release => |button| {
                    switch (button.button) {
                        .left => {
                            self.state.mouse_left = 0;
                        },
                        .right => {
                            self.state.mouse_right = 0;
                        },
                        .middle => {
                            self.state.mouse_middle = 0;
                        },
                        else => {},
                    }
                },
                .mouse_scroll => |del| {
                    self.state.scroll += del.yoffset;
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
                    self.state.render_height = sze.height;
                    self.state.render_width = sze.width;
                    self.state.display_height = sze.height;
                    self.state.display_width = sze.width;
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

        pub fn deinit(self: *@This()) void {
            C.shaderc_result_release(self.result);
        }

        /// owned by Shader
        pub fn bytes(self: *@This()) []const u32 {
            const len = C.shaderc_result_get_length(self.result);
            const res = C.shaderc_result_get_bytes(self.result)[0..len];
            // alignment guranteed when it is spirv
            return @ptrCast(@alignCast(res));
        }
    };

    pub fn testfn() !void {
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
