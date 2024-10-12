const std = @import("std");
const main = @import("main.zig");
const allocator = main.allocator;

pub const Glslc = struct {
    pub fn testfn() !void {
        const compiler = Compiler{
            .opt = .fast,
            .stage = .fragment,
        };
        const shader: []const u8 = @embedFile("frag.glsl");
        const bytes = try compiler.compile(allocator, &.{
            .code = shader,
            .definitions = &[_][]const u8{},
        }, .assembly);
        defer allocator.free(bytes);

        std.debug.print("{s}\n", .{bytes});
    }

    pub const Compiler = struct {
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
        pub const OutputType = enum {
            assembly,
            spirv,
        };
        pub const Code = union(enum) {
            code: struct {
                src: []const u8,
                definitions: []const []const u8,
            },
            path: struct {
                main: []const u8,
                include: []const []const u8,
                definitions: []const []const u8,
            },
        };

        pub fn dump_assembly(self: @This(), alloc: std.mem.Allocator, code: []const u8) !void {
            // std.debug.print("{s}\n", .{code});
            const bytes = try self.compile(alloc, &.{
                .code = code,
                .definitions = &[_][]const u8{},
            }, .assembly);
            defer alloc.free(bytes);
            std.debug.print("{s}\n", .{bytes});
        }

        pub fn compile(self: @This(), alloc: std.mem.Allocator, code: *const Code, comptime output_type: OutputType) !(switch (output_type) {
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
            try args.append(try alloc.dupe(u8, "glslc"));
            try args.append(try alloc.dupe(u8, switch (self.stage) {
                .fragment => "-fshader-stage=fragment",
                .vertex => "-fshader-stage=vertex",
                .compute => "-fshader-stage=compute",
            }));
            try args.append(try alloc.dupe(u8, switch (self.lang) {
                .glsl => "-xglsl",
                .hlsl => "-xhlsl",
            }));
            try args.append(try alloc.dupe(u8, switch (self.opt) {
                .fast => "-O",
                .small => "-Os",
                .none => "-O0",
            }));
            if (output_type == .assembly) {
                try args.append(try alloc.dupe(u8, "-S"));
            }
            try args.append(try alloc.dupe(u8, "-o-"));
            switch (code.*) {
                .code => |src| {
                    for (src.definitions) |def| {
                        try args.append(try std.fmt.allocPrint(alloc, "-D{s}", .{def}));
                    }
                    try args.append(try alloc.dupe(u8, "-"));
                },
                .path => |paths| {
                    for (paths.definitions) |def| {
                        try args.append(try std.fmt.allocPrint(alloc, "-D{s}", .{def}));
                    }
                    for (paths.include) |inc| {
                        try args.append(try alloc.dupe(u8, "-I"));
                        try args.append(try alloc.dupe(u8, inc));
                    }
                    try args.append(try alloc.dupe(u8, paths.main));
                },
            }

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

            switch (code.*) {
                .code => |src| {
                    try stdin.writeAll(src.src);
                },
                .path => {},
            }
            stdin.close();

            // similar to child.collectOutput
            const max_output_bytes = 1000 * 1000;
            var poller = std.io.poll(allocator, enum { stdout, stderr }, .{
                .stdout = stdout,
                .stderr = stderr,
            });
            defer poller.deinit();
            errdefer {
                const fifo = poller.fifo(.stderr);
                std.debug.print("{s}\n", .{fifo.buf[fifo.head..][0..fifo.count]});
            }

            while (try poller.poll()) {
                if (poller.fifo(.stdout).count > max_output_bytes)
                    return error.StdoutStreamTooLong;
                if (poller.fifo(.stderr).count > max_output_bytes)
                    return error.StderrStreamTooLong;
            }

            const err = try child.wait();
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

            const fifo = poller.fifo(.stdout);
            var aligned = std.ArrayListAligned(u8, 4).init(allocator);
            try aligned.appendSlice(fifo.buf[fifo.head..][0..fifo.count]);
            const bytes = try aligned.toOwnedSlice();
            return switch (output_type) {
                .spirv => std.mem.bytesAsSlice(u32, bytes),
                .assembly => bytes,
            };
        }
    };
};

pub const Shaderc = struct {
    const C = @cImport({
        @cInclude("shaderc/shaderc.h");
    });

    pub const Shader = struct {
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

pub const Dawn = struct {
    const C = @cImport({
        @cInclude("dawn/webgpu.h");
    });

    pub fn testfn() !void {}
};

pub const Glslang = struct {
    const C = @cImport({
        @cInclude("glslang/Include/glslang_c_interface.h");
        @cInclude("glslang/Public/resource_limits_c.h");
        @cInclude("stdio.h");
        @cInclude("stdlib.h");
        @cInclude("stdint.h");
        @cInclude("stdbool.h");
    });

    pub const SpirVBinary = struct {
        words: ?*C.u_int32_t, // SPIR-V words
        size: usize, // number of words in SPIR-V binary
    };

    pub fn testfn() !void {
        const shader = @embedFile("shader.glsl");
        const bin = try compileShaderToSPIRV_Vulkan(
            C.GLSLANG_STAGE_VERTEX_MASK | C.GLSLANG_STAGE_FRAGMENT_MASK | C.GLSLANG_STAGE_COMPUTE_MASK,
            shader.ptr,
        );
        _ = bin;
    }

    pub fn compileShaderToSPIRV_Vulkan(stage: C.glslang_stage_t, shaderSource: [*c]const u8) !SpirVBinary {
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
