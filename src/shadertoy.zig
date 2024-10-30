const std = @import("std");

const utils = @import("utils.zig");
const JsonHelpers = utils.JsonHelpers;
const FsFuse = utils.FsFuse;
const Fuse = utils.Fuse;
const Curl = utils.Curl;

const compile = @import("compile.zig");
const Glslc = compile.Glslc;

const main = @import("main.zig");
const allocator = main.allocator;
const config = main.config;

// https://www.shadertoy.com/howto
const base = "https://www.shadertoy.com";
const api_base = base ++ "/api/v1";
const key = "BdHjRn";
const json_parse_ops = .{
    .ignore_unknown_fields = true,
    .allocate = .alloc_always,
};

pub fn testfn() !void {
    return try testfn2();
}

pub fn testfn2() !void {
    const cwd = std.fs.cwd();
    const id = "4td3zj";
    const toys = "./toys/shadertoy";

    var cache = try Cached.init(toys);
    defer cache.deinit();
    var toy = try cache.shader(id);
    defer toy.deinit();
    const cwd_real = try cwd.realpathAlloc(allocator, "./");
    defer allocator.free(cwd_real);
    var man = try ToyMan.init(try std.fs.path.join(allocator, &[_][]const u8{ cwd_real, "shaders" }));
    defer man.deinit();

    const target = try cwd.realpathAlloc(allocator, toys ++ "/" ++ id);
    defer allocator.free(target);
    var ptoy = try man.prepare_toy(&toy.value, target);
    defer ptoy.deinit();
}

pub fn testfn1() !void {
    var cache = try Cached.init("./toys/shadertoy");
    defer cache.deinit();
    // const toy = try cache.shader("4t3SzN");
    const toy = try cache.shader("lXjczd");
    defer toy.deinit();
    // const toy = try Toy.from_file("./toys/new.json");
    // defer toy.deinit();

    std.debug.print("{s}\n", .{toy.value.Shader.renderpass[0].code});
}

pub const ToyMan = struct {
    pub const ShaderMetadata = union(enum) {
        screen,
        image: *ActiveToy.Inputs,
        buffer1: *ActiveToy.Inputs,
        buffer2: *ActiveToy.Inputs,
        buffer3: *ActiveToy.Inputs,
        buffer4: *ActiveToy.Inputs,
    };
    pub const CompiledShader = struct {
        vert: []u32,
        frag: []u32,
    };
    pub const Compiled = struct {
        mutex: std.Thread.Mutex = .{},
        input_fuse: Fuse = .{},
        uniform_reset_fuse: Fuse = .{},
        toy: ActiveToy,
        vert: []u32,

        screen: []u32,
        image: []u32,
        buffer1: ?[]u32 = null,
        buffer2: ?[]u32 = null,
        buffer3: ?[]u32 = null,
        buffer4: ?[]u32 = null,

        fn no_lock_deinit(self: *@This()) void {
            allocator.free(self.vert);
            allocator.free(self.screen);
            allocator.free(self.image);
            if (self.buffer1) |buf| allocator.free(buf);
            if (self.buffer2) |buf| allocator.free(buf);
            if (self.buffer3) |buf| allocator.free(buf);
            if (self.buffer4) |buf| allocator.free(buf);
            self.toy.deinit();
        }

        pub fn deinit(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.no_lock_deinit();
        }
    };
    pub const Compiler = struct {
        const EventChan = utils.Channel(Compiled);
        const ToyChan = utils.Channel(ActiveToy);
        const Ctx = struct {
            config: *main.Renderer.Config,
            shader_fuse: *FsFuse,

            exit: Fuse = .{},
            compile_fuse: Fuse = .{},
            toy_chan: ToyChan,
            compiled: ?Compiled,

            fn try_get_update(self: *@This()) ?UpdateEvent {
                while (self.shader_fuse.try_recv()) |ev| {
                    switch (ev) {
                        .All => {
                            return .All;
                        },
                        .File => |path| {
                            defer allocator.free(path);
                            if (std.mem.eql(u8, path, "buffer1.glsl")) {
                                return .Buffer1;
                            }
                            if (std.mem.eql(u8, path, "buffer2.glsl")) {
                                return .Buffer2;
                            }
                            if (std.mem.eql(u8, path, "buffer3.glsl")) {
                                return .Buffer3;
                            }
                            if (std.mem.eql(u8, path, "buffer4.glsl")) {
                                return .Buffer4;
                            }
                            if (std.mem.eql(u8, path, "image.glsl")) {
                                return .Image;
                            }
                            if (std.mem.eql(u8, path, "common.glsl")) {
                                return .All;
                            }
                            if (std.mem.eql(u8, path, "toy.json")) {
                                return .All;
                            }
                            if (std.mem.eql(u8, path, "frag.glsl")) {
                                return .All;
                            }
                            if (std.mem.eql(u8, path, "vert.glsl")) {
                                return .All;
                            }
                            std.debug.print("Unknown file update: {s}\n", .{path});
                        },
                    }
                }
                return null;
            }

            fn handle_update_blocked(self: *@This(), e: UpdateEvent, at: *ActiveToy) !void {
                switch (e) {
                    .Buffer1 => {
                        self.compiled.?.mutex.lock();
                        defer self.compiled.?.mutex.unlock();

                        if (self.compiled.?.buffer1) |buf| {
                            self.compiled.?.buffer1 = try self.compile(.{
                                .buffer1 = &at.passes.buffer1.?.inputs,
                            }, at.has_common);
                            allocator.free(buf);
                        }
                    },
                    .Buffer2 => {
                        self.compiled.?.mutex.lock();
                        defer self.compiled.?.mutex.unlock();

                        if (self.compiled.?.buffer2) |buf| {
                            self.compiled.?.buffer2 = try self.compile(.{
                                .buffer2 = &at.passes.buffer2.?.inputs,
                            }, at.has_common);
                            allocator.free(buf);
                        }
                    },
                    .Buffer3 => {
                        self.compiled.?.mutex.lock();
                        defer self.compiled.?.mutex.unlock();

                        if (self.compiled.?.buffer3) |buf| {
                            self.compiled.?.buffer3 = try self.compile(.{
                                .buffer3 = &at.passes.buffer3.?.inputs,
                            }, at.has_common);
                            allocator.free(buf);
                        }
                    },
                    .Buffer4 => {
                        self.compiled.?.mutex.lock();
                        defer self.compiled.?.mutex.unlock();

                        if (self.compiled.?.buffer4) |buf| {
                            self.compiled.?.buffer4 = try self.compile(.{
                                .buffer4 = &at.passes.buffer4.?.inputs,
                            }, at.has_common);
                            allocator.free(buf);
                        }
                    },
                    .Image => {
                        self.compiled.?.mutex.lock();
                        defer self.compiled.?.mutex.unlock();

                        const buf = self.compiled.?.image;
                        self.compiled.?.image = try self.compile(.{
                            .image = &at.passes.image.inputs,
                        }, at.has_common);
                        allocator.free(buf);
                    },
                    .All => {
                        var compiled = blk: {
                            const image = try self.compile(.{
                                .image = &at.passes.image.inputs,
                            }, at.has_common);
                            errdefer allocator.free(image);

                            const screen = try self.compile(.screen, at.has_common);
                            errdefer allocator.free(screen);

                            const vert = try self.compile_vert();
                            errdefer allocator.free(vert);

                            break :blk Compiled{
                                .vert = vert,
                                .screen = screen,
                                .image = image,
                                .toy = try at.clone(),
                            };
                        };
                        errdefer compiled.deinit();

                        if (at.passes.buffer1) |_| compiled.buffer1 = try self.compile(.{
                            .buffer1 = &at.passes.buffer1.?.inputs,
                        }, at.has_common);
                        if (at.passes.buffer2) |_| compiled.buffer2 = try self.compile(.{
                            .buffer2 = &at.passes.buffer2.?.inputs,
                        }, at.has_common);
                        if (at.passes.buffer3) |_| compiled.buffer3 = try self.compile(.{
                            .buffer3 = &at.passes.buffer3.?.inputs,
                        }, at.has_common);
                        if (at.passes.buffer4) |_| compiled.buffer4 = try self.compile(.{
                            .buffer4 = &at.passes.buffer4.?.inputs,
                        }, at.has_common);

                        _ = compiled.input_fuse.fuse();
                        _ = compiled.uniform_reset_fuse.fuse();

                        if (self.compiled) |*comp| {
                            comp.mutex.lock();
                            defer comp.mutex.unlock();
                            comp.no_lock_deinit();

                            compiled.mutex = comp.mutex;
                            comp.* = compiled;
                        } else {
                            self.compiled = compiled;
                        }
                    },
                }

                _ = self.compile_fuse.fuse();
            }

            pub fn compile(self: *@This(), meta: ShaderMetadata, has_common: bool) ![]u32 {
                return try compile_frag(self.config, has_common, meta);
            }

            pub fn compile_vert(self: *@This()) ![]u32 {
                return try Compiler.compile_vert(self.config);
            }

            pub fn reload_all(self: *@This()) !void {
                try self.shader_fuse.ctx.channel.send(.All);
            }

            fn deinit(self: *@This()) void {
                while (self.toy_chan.try_recv()) |ev_| {
                    var ev = ev_;
                    ev.deinit();
                }
                self.toy_chan.deinit();
                if (self.compiled) |*c| c.deinit();

                // NOTE: not owned
                // self.config
                // self.shader_fuse
            }
        };
        ctx: *Ctx,
        thread: std.Thread,

        pub fn init(render_config: *main.Renderer.Config, shader_fuse: *FsFuse) !@This() {
            var toy_chan = try ToyChan.init(allocator);
            errdefer toy_chan.deinit();

            const ctxt = try allocator.create(Ctx);
            errdefer allocator.destroy(ctxt);
            ctxt.* = .{
                .toy_chan = toy_chan,
                .config = render_config,
                .shader_fuse = shader_fuse,
                .compiled = null,
            };
            errdefer ctxt.deinit();

            const Callbacks = struct {
                fn spawn(ctx: *Ctx) void {
                    var at = ActiveToy{};
                    defer at.deinit();

                    while (true) {
                        if (ctx.exit.unfuse()) {
                            break;
                        }
                        while (ctx.toy_chan.try_recv()) |toy| {
                            at.deinit();
                            at = toy;
                        }
                        while (ctx.try_get_update()) |e| ctx.handle_update_blocked(e, &at) catch continue;

                        std.time.sleep(std.time.ns_per_ms * 100);
                    }
                }
            };
            const thread = try std.Thread.spawn(.{ .allocator = allocator }, Callbacks.spawn, .{ctxt});

            return .{
                .ctx = ctxt,
                .thread = thread,
            };
        }

        pub fn deinit(self: *@This()) void {
            _ = self.ctx.exit.fuse();
            self.thread.join();
            self.ctx.deinit();
            allocator.destroy(self.ctx);
        }

        pub fn fuse_resize(self: *@This()) void {
            if (self.ctx.compiled) |*comp| {
                comp.mutex.lock();
                defer comp.mutex.unlock();

                _ = comp.input_fuse.fuse();
            }
            _ = self.ctx.compile_fuse.fuse();
        }

        pub fn unfuse(self: *@This()) ?*Compiled {
            if (self.ctx.compile_fuse.unfuse()) {
                if (self.ctx.compiled) |*compiled| {
                    return compiled;
                } else {
                    return null;
                }
            }
            return null;
        }

        fn compile_vert(
            render_config: *const main.Renderer.Config,
        ) ![]u32 {
            var compiler = Glslc.Compiler{ .opt = render_config.shader_compile_opt };
            compiler.stage = .vertex;
            const vert: Glslc.Compiler.Code = .{ .path = .{
                .main = "./shaders/vert.glsl",
                .include = &[_][]const u8{config.paths.playground},
                .definitions = &[_][]const u8{},
            } };
            // if (render_config.shader_dump_assembly) {
            //     _ = compiler.dump_assembly(allocator, &vert);
            // }
            const vertex_bytes = blk: {
                const res = try compiler.compile(
                    allocator,
                    &vert,
                    .spirv,
                );
                switch (res) {
                    .Err => |msg| {
                        try render_config.gpu_err_fuse.err_channel.send(msg.msg);
                        return msg.err;
                    },
                    .Ok => |ok| {
                        errdefer allocator.free(ok);
                        try render_config.gpu_err_fuse.err_channel.send(null);
                        break :blk ok;
                    },
                }
            };
            return vertex_bytes;
        }

        fn compile_frag(
            render_config: *const main.Renderer.Config,
            has_common: bool,
            meta: ShaderMetadata,
        ) ![]u32 {
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

            switch (meta) {
                .screen => {},
                .image, .buffer1, .buffer2, .buffer3, .buffer4 => {
                    try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL0"));
                    try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL1"));
                    try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL2"));
                    try definitions.append(try allocator.dupe(u8, "ZHADER_CHANNEL3"));
                },
            }
            switch (meta) {
                .screen => {},
                .image, .buffer1, .buffer2, .buffer3, .buffer4 => |inputs| {
                    for ([_]*const ?ActiveToy.Input{ &inputs.input1, &inputs.input2, &inputs.input3, &inputs.input4 }, 0..) |maybe_inp, i| {
                        var typ: enum {
                            d2,
                            cubemap,
                        } = .d2;
                        if (maybe_inp.*) |inp| {
                            switch (inp.typ) {
                                .keyboard, .mic, .webcam, .texture => {
                                    typ = .d2;
                                },

                                .writable => |w| switch (w) {
                                    .Cubemap => {
                                        typ = .cubemap;
                                    },
                                    .BufferA, .BufferB, .BufferC, .BufferD => {
                                        typ = .d2;
                                    },
                                },
                                .cubemap => {
                                    typ = .cubemap;
                                },

                                // TODO:
                                .volume, .video, .music => {},
                            }
                        }

                        switch (typ) {
                            .d2 => {
                                try definitions.append(try std.fmt.allocPrint(allocator, "ZHADER_CHANNEL{d}_2D", .{i}));
                            },
                            .cubemap => {
                                try definitions.append(try std.fmt.allocPrint(allocator, "ZHADER_CHANNEL{d}_CUBEMAP", .{i}));
                            },
                        }
                    }
                },
            }

            // TODO: cubemap pass not implemented.
            // i can't find many cubemap pass shaders to test :/
            try definitions.append(try std.fmt.allocPrint(allocator, "ZHADER_INCLUDE_{s}", .{switch (meta) {
                .screen => "SCREEN",
                .image => "IMAGE",
                .buffer1 => "BUF1",
                .buffer2 => "BUF2",
                .buffer3 => "BUF3",
                .buffer4 => "BUF4",
            }}));

            var compiler = Glslc.Compiler{ .opt = render_config.shader_compile_opt };
            compiler.stage = .fragment;
            const frag: Glslc.Compiler.Code = .{ .path = .{
                .main = "./shaders/frag.glsl",
                .include = &[_][]const u8{config.paths.playground},
                .definitions = definitions.items,
            } };
            if (render_config.shader_dump_assembly) blk: {
                // TODO: print this on screen instead of console
                const res = compiler.dump_assembly(allocator, &frag) catch {
                    break :blk;
                };
                switch (res) {
                    .Err => |err| {
                        try render_config.gpu_err_fuse.err_channel.send(err.msg);
                        return err.err;
                    },
                    .Ok => {
                        try render_config.gpu_err_fuse.err_channel.send(null);
                    },
                }
            }
            const frag_bytes = blk: {
                const res = try compiler.compile(
                    allocator,
                    &frag,
                    .spirv,
                );
                switch (res) {
                    .Err => |err| {
                        try render_config.gpu_err_fuse.err_channel.send(err.msg);
                        return err.err;
                    },
                    .Ok => |ok| {
                        errdefer allocator.free(ok);
                        try render_config.gpu_err_fuse.err_channel.send(null);
                        break :blk ok;
                    },
                }
            };
            return frag_bytes;
        }
    };

    pub const UpdateEvent = enum {
        Buffer1,
        Buffer2,
        Buffer3,
        Buffer4,
        Image,
        All,
    };

    // generate glsl glue code for shadertoy to work properly with vulkan glsl
    //
    // struct ZhaderChannel2D { int id; };
    // ZhaderChannel2D iChannel0 = {0};
    // ZhaderChannel2D iChannel2 = {2};
    // texture(ZhaderChannel2D chan) {
    //   if (chan.id == 0) {
    //     return texture(zhader_channel0);
    //   }
    //   if (chan.id == 2) {
    //     return texture(zhader_channel2);
    //   }
    //
    //   return vec4(0.0);
    // }
    // #define sampler2D ZhaderChannel2D
    pub const Generator = struct {
        channels: [4]Typ,

        pub fn testfn() !void {
            var gen: @This() = .{
                .channels = [_]Typ{ .d2, .d2, .d2, .d2 },
            };

            try gen.generate(2);
        }

        fn from(inputs: *const ActiveToy.Inputs) @This() {
            var gen: @This() = .{
                .channels = [_]Typ{ .d2, .d2, .d2, .d2 },
            };

            if (inputs.input1) |inp| {
                gen.channels[0] = Typ.from(&inp.typ);
            }
            if (inputs.input2) |inp| {
                gen.channels[1] = Typ.from(&inp.typ);
            }
            if (inputs.input3) |inp| {
                gen.channels[2] = Typ.from(&inp.typ);
            }
            if (inputs.input4) |inp| {
                gen.channels[3] = Typ.from(&inp.typ);
            }

            return gen;
        }

        var arena: std.mem.Allocator = undefined;

        const Writer = std.ArrayList(u8).Writer;

        const Typ = enum {
            d2,
            d3,
            cube,

            fn from(t: *const ActiveToy.Channel) @This() {
                return switch (t.*) {
                    .writable => |w| switch (w) {
                        .Cubemap => .cube,
                        .BufferA, .BufferB, .BufferC, .BufferD => .d2,
                    },
                    .keyboard, .mic, .webcam, .texture => .d2,
                    .cubemap => .cube,

                    // TODO:
                    .volume, .video, .music => .d2,
                };
            }

            fn name(self: @This()) []const u8 {
                return switch (self) {
                    .d2 => "2D",
                    .d3 => "3D",
                    .cube => "Cube",
                };
            }

            fn texture_coord_type(self: @This()) []const u8 {
                return switch (self) {
                    .d2 => "vec2",
                    .d3 => "vec3",
                    .cube => "vec3",
                };
            }

            fn struct_name(self: @This()) []const u8 {
                return switch (self) {
                    .d2 => "ZhaderChannel2D",
                    .d3 => "ZhaderChannel3D",
                    .cube => "ZhaderChannelCube",
                };
            }

            fn struct_decl(self: @This(), w: Writer) !void {
                try w.print(
                    \\ struct {s} {{
                    \\     int id;
                    \\ }};
                    \\
                , .{self.struct_name()});
            }

            fn binding(self: @This(), w: Writer, chan: u32, i: u32, j: u32) !void {
                try w.print(
                    \\ layout(set = 0, binding = {d}) uniform texture{s} zhader_channel_{d};
                    \\ layout(set = 0, binding = {d}) uniform sampler zhader_sampler_{d};
                    \\
                , .{ i, self.name(), chan, j, chan });
            }

            fn type_define(self: @This(), w: Writer) !void {
                try w.print(
                    \\ #define sampler{s} {s}
                    \\
                , .{ self.name(), self.struct_name() });
            }

            fn channel_var(self: @This(), w: Writer, chan: u32) !void {
                try w.print(
                    \\ {s} iChannel{d} = {{ {d} }};
                    \\
                , .{ self.struct_name(), chan, chan });
            }

            fn function(self: @This(), w: Writer, channels: [4]@This(), vflip: enum { none, float, int }, func: Func, args: []const Arg, ret: ReturnType) !void {
                try w.print(" {s} {s}({s} chan", .{ @tagName(ret), @tagName(func), self.struct_name() });
                for (args) |arg| {
                    try w.print(", {s}", .{arg.decl()});
                }
                try w.print(") {{\n", .{});

                for (channels, 0..) |chan, i| {
                    if (self == chan) {
                        try w.print(
                            \\     if (chan.id == {d}) {{
                            \\
                        , .{i});

                        switch (vflip) {
                            .none => {},
                            .float => {
                                try w.print(
                                    \\         if (vflips[{d}] == 1) {{
                                    \\             pos.y = 1.0 - pos.y;
                                    \\         }}
                                    \\
                                    \\
                                , .{i});
                            },
                            .int => {
                                try w.print(
                                    \\         if (vflips[{d}] == 1) {{
                                    \\             pos.y = textureSize(chan, lod).y - pos.y - 1;
                                    \\         }}
                                    \\
                                    \\
                                , .{i});
                            },
                        }

                        try w.print(
                            \\         return {s}(sampler{s}(zhader_channel_{d}, zhader_sampler_{d})
                        , .{ @tagName(func), self.name(), i, i });

                        for (args) |arg| {
                            try w.print(", {s}", .{arg.name()});
                        }

                        try w.print(");\n", .{});
                        try w.print(
                            \\     }}
                            \\
                        , .{});
                    }
                }

                try w.print(
                    \\
                    \\     return {s}(0);
                    \\ }}
                    \\
                , .{@tagName(ret)});
            }
        };
        const Arg = enum {
            vec2pos,
            ivec2pos,
            vec3pos,
            vec4pos,
            vec2dpdx,
            vec2dpdy,
            intlod,
            floatlod,

            fn typ(self: @This()) []const u8 {
                return switch (self) {
                    .vec2pos => "vec2",
                    .ivec2pos => "ivec2",
                    .vec3pos => "vec3",
                    .vec4pos => "vec4",
                    .vec2dpdx => "vec2",
                    .vec2dpdy => "vec2",
                    .intlod => "int",
                    .floatlod => "float",
                };
            }

            fn name(self: @This()) []const u8 {
                return switch (self) {
                    .vec2pos => "pos",
                    .ivec2pos => "pos",
                    .vec3pos => "pos",
                    .vec4pos => "pos",
                    .vec2dpdx => "dpdx",
                    .vec2dpdy => "dpdy",
                    .intlod => "lod",
                    .floatlod => "lod",
                };
            }

            fn decl(self: @This()) []const u8 {
                return switch (self) {
                    .vec2pos => "vec2 pos",
                    .ivec2pos => "ivec2 pos",
                    .vec3pos => "vec3 pos",
                    .vec4pos => "vec4 pos",
                    .vec2dpdx => "vec2 dpdx",
                    .vec2dpdy => "vec2 dpdy",
                    .intlod => "int lod",
                    .floatlod => "float lod",
                };
            }
        };
        const Func = enum {
            texture,
            textureSize,
            texelFetch,
            textureQueryLevels,
            textureLod,
            textureProj,
            textureGather,
            textureGrad,
            textureQueryLod,
        };
        const ReturnType = enum {
            vec4,
            ivec4,
            ivec2,
            vec2,
            int,
        };

        fn generate(self: *@This(), binding_start: u32) ![]const u8 {
            var code = std.ArrayList(u8).init(allocator);
            errdefer code.deinit();
            const w = code.writer();

            try Typ.d2.struct_decl(w);
            try Typ.d3.struct_decl(w);
            try Typ.cube.struct_decl(w);

            var binding: u32 = binding_start;
            for (self.channels, 0..) |chan, i| {
                defer binding += 2;
                try chan.binding(w, @intCast(i), binding, binding + 1);
                try chan.channel_var(w, @intCast(i));
            }

            const c = self.channels;
            try Typ.d2.function(w, c, .float, .texture, &[_]Arg{.vec2pos}, .vec4);
            try Typ.cube.function(w, c, .float, .texture, &[_]Arg{.vec3pos}, .vec4);
            try Typ.d2.function(w, c, .float, .textureProj, &[_]Arg{.vec4pos}, .vec4);
            try Typ.d2.function(w, c, .none, .textureSize, &[_]Arg{.intlod}, .ivec2);
            try Typ.d2.function(w, c, .int, .texelFetch, &[_]Arg{ .ivec2pos, .intlod }, .vec4);
            try Typ.d2.function(w, c, .float, .textureLod, &[_]Arg{ .vec2pos, .floatlod }, .vec4);
            try Typ.d2.function(w, c, .float, .textureGrad, &[_]Arg{ .vec2pos, .vec2dpdx, .vec2dpdy }, .vec4);
            try Typ.d2.function(w, c, .float, .textureGather, &[_]Arg{.vec2pos}, .vec4);
            try Typ.d2.function(w, c, .none, .textureQueryLevels, &[_]Arg{}, .int);
            try Typ.d2.function(w, c, .float, .textureQueryLod, &[_]Arg{.vec2pos}, .vec2);

            try Typ.d2.type_define(w);
            try Typ.d3.type_define(w);
            try Typ.cube.type_define(w);

            return try code.toOwnedSlice();
        }
    };

    // path where the toy should be simlinked
    playground: []const u8,
    shader_fuse: *FsFuse,
    shader_cache: Cached,
    active_toy: *ActiveToy,
    compiler: Compiler,

    const vert_glsl = @embedFile("./vert.glsl");
    const frag_glsl = @embedFile("./frag.glsl");

    pub fn init(render_config: *main.Renderer.Config) !@This() {
        const cwd_real = try std.fs.cwd().realpathAlloc(allocator, "./");
        defer allocator.free(cwd_real);
        const playground = try std.fs.path.join(allocator, &[_][]const u8{ cwd_real, "shaders" });

        var fuse = try allocator.create(FsFuse);
        errdefer allocator.destroy(fuse);
        fuse.* = try FsFuse.init(config.paths.playground);
        errdefer fuse.deinit();
        // TODO: fuse() + default self.active_toy won't work cuz of things like has_common
        try fuse.ctx.channel.send(.All);
        var cache = try Cached.init(config.paths.shadertoys);
        errdefer cache.deinit();

        const at = try allocator.create(ActiveToy);
        at.* = .{};
        errdefer allocator.destroy(at);

        const compiler = try Compiler.init(render_config, fuse);

        return .{
            .playground = playground,
            .shader_fuse = fuse,
            .shader_cache = cache,
            .active_toy = at,
            .compiler = compiler,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.compiler.deinit();
        self.shader_fuse.deinit();
        self.shader_cache.deinit();
        self.active_toy.deinit();
        allocator.destroy(self.active_toy);
        allocator.free(self.playground);
        allocator.destroy(self.shader_fuse);
    }

    pub fn has_updates(self: *@This()) bool {
        return self.compiler.ctx.compile_fuse.check();
    }

    pub fn load_shadertoy(self: *@This(), id: []const u8) !void {
        var toy = try self.shader_cache.shader(id);
        defer toy.deinit();

        const path = try std.fs.path.join(allocator, &[_][]const u8{ config.paths.shadertoys, id });
        defer allocator.free(path);
        const target = try std.fs.cwd().realpathAlloc(allocator, path);
        defer allocator.free(target);

        const active = try prepare_toy(&self.shader_cache, self.playground, &toy.value, target, false);
        self.active_toy.deinit();
        self.active_toy.* = active;
        try self.compiler.ctx.toy_chan.send(try active.clone());

        try self.shader_fuse.restart(config.paths.playground);
        try self.shader_fuse.ctx.channel.send(.All);
    }

    pub fn load_zhadertoy(self: *@This(), id: []const u8) !void {
        const path = try std.fs.path.join(allocator, &[_][]const u8{ config.paths.zhadertoys, id });
        defer allocator.free(path);
        const target = try std.fs.cwd().realpathAlloc(allocator, path);
        defer allocator.free(target);

        const toypath = try std.fs.path.join(allocator, &[_][]const u8{ path, "toy.json" });
        defer allocator.free(toypath);
        var toy = try Toy.from_file(toypath);
        defer toy.deinit();

        const active = try prepare_toy(&self.shader_cache, self.playground, &toy.value, target, false);
        self.active_toy.deinit();
        self.active_toy.* = active;
        try self.compiler.ctx.toy_chan.send(try active.clone());

        try self.shader_fuse.restart(config.paths.playground);
        try self.shader_fuse.ctx.channel.send(.All);
    }

    pub fn reset_toy(self: *@This()) !void {
        const toypath = try std.fs.path.join(allocator, &[_][]const u8{ self.playground, "toy.json" });
        defer allocator.free(toypath);
        var toy = try Toy.from_file(toypath);
        defer toy.deinit();

        const active = try prepare_toy(&self.shader_cache, self.playground, &toy.value, self.playground, true);
        self.active_toy.deinit();
        self.active_toy.* = active;
        try self.compiler.ctx.toy_chan.send(try active.clone());

        try self.shader_fuse.ctx.channel.send(.All);
    }

    // file naming:
    //   - virt.glsl: vertex shader entry point (not available on shadertoy)
    //   - frag.glsl: fragment shader entry point (not available on shadertoy)
    //   - common.glsl
    //   - image.glsl
    //   - sound.glsl
    //   - cube.glsl
    //   - buffer1A.glsl: where the '1' represents execution order of the buffers
    //       (shadertoy buffers ABCD can be in any order, but they execute in the order they are defined in)
    pub fn prepare_toy(cache: *Cached, playground: []const u8, toy: *Toy, dir_path: []const u8, reset: bool) !ActiveToy {
        var dir = try std.fs.openDirAbsolute(dir_path, .{});
        defer dir.close();

        // testing if frag exists to figure out if we need to prep or load the toy
        const prep = blk: {
            dir.access("frag.glsl", .{}) catch break :blk true;
            break :blk reset;
        };

        var t = ActiveToy{};

        for (toy.Shader.renderpass) |pass| {
            switch (pass.type) {
                .common => {
                    t.has_common = true;

                    if (prep) {
                        var buf = try dir.createFile("common.glsl", .{});
                        defer buf.close();

                        try buf.writeAll(pass.code);
                    }
                },
                .buffer => {
                    if (pass.outputs.len != 1) return error.BadBufferPassOutput;
                    const out_id = pass.outputs[0].id;
                    const out_buf = try std.meta.intToEnum(ActiveToy.Writable, out_id);

                    var num: u32 = 1;
                    var buffer = &t.passes.buffer1;
                    if (t.passes.buffer1) |_| {
                        num += 1;
                        buffer = &t.passes.buffer2;
                    }
                    if (t.passes.buffer2) |_| {
                        num += 1;
                        buffer = &t.passes.buffer3;
                    }
                    if (t.passes.buffer3) |_| {
                        num += 1;
                        buffer = &t.passes.buffer4;
                    }
                    if (t.passes.buffer4) |_| return error.TooManyBufferPasses;

                    const name = try std.fmt.allocPrint(
                        allocator,
                        // "buffer{d}{c}.glsl", // TODO:
                        // .{ num, pass.name[pass.name.len - 1] },
                        "buffer{d}.glsl",
                        .{num},
                    );
                    defer allocator.free(name);

                    if (prep) {
                        var buf = try dir.createFile(name, .{});
                        defer buf.close();

                        try buf.writeAll(pass.code);
                    }

                    var inputs = try ActiveToy.Inputs.from(pass.inputs, cache);
                    errdefer inputs.deinit();

                    var gen = Generator.from(&inputs);
                    const generated = try gen.generate(2);
                    defer allocator.free(generated);

                    const gen_name = try std.fmt.allocPrint(
                        allocator,
                        "generated_{s}",
                        .{name},
                    );
                    defer allocator.free(gen_name);

                    if (prep) {
                        var buf = try dir.createFile(gen_name, .{});
                        defer buf.close();

                        try buf.writeAll(generated);
                    }

                    buffer.* = .{
                        .output = out_buf,
                        .inputs = inputs,
                    };
                },
                .image => {
                    if (prep) {
                        var buf = try dir.createFile("image.glsl", .{});
                        defer buf.close();

                        try buf.writeAll(pass.code);
                    }

                    var inputs = try ActiveToy.Inputs.from(pass.inputs, cache);
                    errdefer inputs.deinit();

                    var gen = Generator.from(&inputs);
                    const generated = try gen.generate(2);
                    defer allocator.free(generated);

                    if (prep) {
                        var buf = try dir.createFile("generated_image.glsl", .{});
                        defer buf.close();

                        try buf.writeAll(generated);
                    }

                    t.passes.image.inputs = inputs;
                },
                else => {
                    std.debug.print("unimplemented renderpass type: {any}\n", .{pass.type});
                },
            }
        }
        t.mark_current_frame_inputs();

        if (prep) {
            var vert = try dir.createFile("vert.glsl", .{});
            defer vert.close();
            try vert.writeAll(vert_glsl);

            var frag = try dir.createFile("frag.glsl", .{});
            defer frag.close();
            try frag.writeAll(frag_glsl);
        }

        if (!std.mem.eql(u8, playground, dir_path)) {
            // ignore if path does not exist.
            dir.deleteFile(playground) catch |e| {
                switch (e) {
                    error.FileNotFound => {},
                    else => {
                        return e;
                    },
                }
            };
            try std.fs.symLinkAbsolute(
                dir_path,
                playground,
                .{ .is_directory = true },
            );
        }

        return t;
    }
};

// what do i need to know? (for now, only consider buffers)
//  - pass
//    - output: buffer{1, 2, 3, 4}
//    - 4 inputs: buffer{1, 2, 3, 4}
pub const ActiveToy = struct {
    pub const Writable = enum(u32) {
        BufferA = 257,
        BufferB = 258,
        BufferC = 259,
        BufferD = 260,
        Cubemap = 41,
    };
    pub const Channel = union(enum) {
        writable: Writable,
        keyboard,
        mic,
        webcam,
        texture: struct {
            name: [:0]const u8,
            img: utils.ImageMagick.FloatImage,
        },
        cubemap: struct {
            name: [:0]const u8,
            img0: utils.ImageMagick.FloatImage,
            img1: utils.ImageMagick.FloatImage,
            img2: utils.ImageMagick.FloatImage,
            img3: utils.ImageMagick.FloatImage,
            img4: utils.ImageMagick.FloatImage,
            img5: utils.ImageMagick.FloatImage,
        },

        // TODO:
        volume,
        video,
        music,

        pub fn deinit(self: *@This()) void {
            switch (self.*) {
                .texture => |*tex| {
                    allocator.free(tex.name);
                    tex.img.deinit();
                },
                .cubemap => |*cube| {
                    allocator.free(cube.name);
                    cube.img0.deinit();
                    cube.img1.deinit();
                    cube.img2.deinit();
                    cube.img3.deinit();
                    cube.img4.deinit();
                    cube.img5.deinit();
                },
                else => {},
            }
        }
    };
    pub const Sampler = Toy.Sampler;
    pub const Input = struct {
        typ: Channel,
        sampler: Sampler,
        from_current_frame: bool = false,

        fn from(input: Toy.Input, cache: *Cached) !@This() {
            switch (input.ctype) {
                .buffer => {
                    return .{
                        .typ = .{ .writable = try std.meta.intToEnum(Writable, input.id) },
                        .sampler = input.sampler,
                    };
                },
                .texture => {
                    var img = try cache.media_img(input.src);
                    errdefer img.deinit();
                    const name = try allocator.dupeZ(u8, input.src);
                    errdefer allocator.free(name);
                    return .{ .typ = .{
                        .texture = .{
                            .img = img,
                            .name = name,
                        },
                    }, .sampler = input.sampler };
                },
                .keyboard => {
                    return .{ .typ = .keyboard, .sampler = input.sampler };
                },
                .mic => {
                    return .{ .typ = .mic, .sampler = input.sampler };
                },
                .webcam => {
                    return .{ .typ = .webcam, .sampler = input.sampler };
                },
                .cubemap => {
                    if (@intFromEnum(Writable.Cubemap) == input.id) {
                        return .{ .typ = .{ .writable = .Cubemap }, .sampler = input.sampler };
                    }

                    var media_src = std.ArrayList(u8).init(allocator);
                    defer media_src.deinit();
                    try media_src.appendSlice(input.src);

                    var img0 = try cache.media_img(media_src.items);
                    errdefer img0.deinit();

                    const ext = std.fs.path.extension(input.src);
                    media_src.items.len -= ext.len;
                    try media_src.appendSlice("_1");
                    try media_src.appendSlice(ext);
                    var img1 = try cache.media_img(media_src.items);
                    errdefer img1.deinit();

                    media_src.items[media_src.items.len - 1 - ext.len] += 1;
                    var img2 = try cache.media_img(media_src.items);
                    errdefer img2.deinit();

                    media_src.items[media_src.items.len - 1 - ext.len] += 1;
                    var img3 = try cache.media_img(media_src.items);
                    errdefer img3.deinit();

                    media_src.items[media_src.items.len - 1 - ext.len] += 1;
                    var img4 = try cache.media_img(media_src.items);
                    errdefer img4.deinit();

                    media_src.items[media_src.items.len - 1 - ext.len] += 1;
                    var img5 = try cache.media_img(media_src.items);
                    errdefer img5.deinit();

                    const name = try allocator.dupeZ(u8, input.src);
                    errdefer allocator.free(name);
                    return .{
                        .typ = .{
                            .cubemap = .{
                                .img0 = img0,
                                .img1 = img1,
                                .img2 = img2,
                                .img3 = img3,
                                .img4 = img4,
                                .img5 = img5,
                                .name = name,
                            },
                        },
                        .sampler = input.sampler,
                    };
                },
                .music, .video, .volume => {
                    // TODO:
                    std.debug.print("did not handle ctype '{any}'\n", .{input.ctype});
                    return .{ .typ = .keyboard, .sampler = input.sampler };
                },
            }
        }

        fn mark_current_frame_input(self: *@This(), typ: Writable) void {
            if (std.meta.eql(self.typ, .{ .writable = typ })) {
                self.from_current_frame = true;
            }
        }

        fn clone(self: *const @This()) !@This() {
            var this = self.*;
            switch (this.typ) {
                .texture => |*tex| {
                    tex.img.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), tex.img.buffer);
                    tex.name = try allocator.dupeZ(u8, tex.name);
                },
                .cubemap => |*cube| {
                    cube.img0.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img0.buffer);
                    cube.img1.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img1.buffer);
                    cube.img2.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img2.buffer);
                    cube.img3.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img3.buffer);
                    cube.img4.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img4.buffer);
                    cube.img5.buffer = try allocator.dupe(utils.ImageMagick.Pixel(f32), cube.img5.buffer);
                    cube.name = try allocator.dupeZ(u8, cube.name);
                },
                else => {},
            }
            return this;
        }

        fn deinit(self: *@This()) void {
            self.typ.deinit();
        }
    };
    pub const Inputs = struct {
        input1: ?Input = null,
        input2: ?Input = null,
        input3: ?Input = null,
        input4: ?Input = null,

        fn from(inputs: []Toy.Input, cache: *Cached) !@This() {
            var self: @This() = .{};
            for (inputs) |input| {
                switch (input.channel) {
                    0 => {
                        self.input1 = try Input.from(input, cache);
                    },
                    1 => {
                        self.input2 = try Input.from(input, cache);
                    },
                    2 => {
                        self.input3 = try Input.from(input, cache);
                    },
                    3 => {
                        self.input4 = try Input.from(input, cache);
                    },
                }
            }
            return self;
        }

        fn mark_current_frame_input(self: *@This(), typ: Writable) void {
            if (self.input1) |*inp| inp.mark_current_frame_input(typ);
            if (self.input2) |*inp| inp.mark_current_frame_input(typ);
            if (self.input3) |*inp| inp.mark_current_frame_input(typ);
            if (self.input4) |*inp| inp.mark_current_frame_input(typ);
        }

        fn clone(self: *const @This()) !@This() {
            var this = self.*;
            this.input1 = null;
            this.input2 = null;
            this.input3 = null;
            this.input4 = null;
            errdefer this.deinit();

            if (self.input1) |inp| {
                this.input1 = try inp.clone();
            }
            if (self.input2) |inp| {
                this.input2 = try inp.clone();
            }
            if (self.input3) |inp| {
                this.input3 = try inp.clone();
            }
            if (self.input4) |inp| {
                this.input4 = try inp.clone();
            }
            return this;
        }

        fn deinit(self: *@This()) void {
            if (self.input1) |*inp| inp.deinit();
            if (self.input2) |*inp| inp.deinit();
            if (self.input3) |*inp| inp.deinit();
            if (self.input4) |*inp| inp.deinit();
        }
    };
    pub const BufferPass = struct {
        output: Writable,
        inputs: Inputs,

        fn clone(self: *const @This()) !@This() {
            var this = self.*;
            this.inputs = try self.inputs.clone();
            return this;
        }

        fn deinit(self: *@This()) void {
            self.inputs.deinit();
        }
    };

    has_common: bool = false,
    passes: struct {
        image: struct {
            inputs: Inputs = .{},

            fn deinit(self: *@This()) void {
                self.inputs.deinit();
            }
        } = .{},
        buffer1: ?BufferPass = null,
        buffer2: ?BufferPass = null,
        buffer3: ?BufferPass = null,
        buffer4: ?BufferPass = null,
    } = .{},

    // any pass can take input of any buffer.
    // any buffers output from 1 pass go into the next passes
    // buffer passes can take input of the same buffer from last frames output
    fn mark_current_frame_inputs(self: *@This()) void {
        if (self.passes.buffer1) |*buf| {
            if (self.passes.buffer2) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            if (self.passes.buffer3) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            self.passes.image.inputs.mark_current_frame_input(buf.output);
        }
        if (self.passes.buffer2) |*buf| {
            if (self.passes.buffer3) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            self.passes.image.inputs.mark_current_frame_input(buf.output);
        }
        if (self.passes.buffer3) |*buf| {
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_input(buf.output);
            }
            self.passes.image.inputs.mark_current_frame_input(buf.output);
        }
        if (self.passes.buffer4) |*buf| {
            self.passes.image.inputs.mark_current_frame_input(buf.output);
        }
    }

    pub fn clone(self: *const @This()) !@This() {
        var this = self.*;
        this.passes = .{};
        errdefer this.deinit();

        this.passes.image.inputs = try self.passes.image.inputs.clone();
        if (self.passes.buffer1) |pass| {
            this.passes.buffer1 = try pass.clone();
        }
        if (self.passes.buffer2) |pass| {
            this.passes.buffer2 = try pass.clone();
        }
        if (self.passes.buffer3) |pass| {
            this.passes.buffer3 = try pass.clone();
        }
        if (self.passes.buffer4) |pass| {
            this.passes.buffer4 = try pass.clone();
        }

        return this;
    }

    pub fn deinit(self: *@This()) void {
        self.passes.image.deinit();
        if (self.passes.buffer1) |*buf| buf.deinit();
        if (self.passes.buffer2) |*buf| buf.deinit();
        if (self.passes.buffer3) |*buf| buf.deinit();
        if (self.passes.buffer4) |*buf| buf.deinit();
    }
};

pub const Toy = struct {
    pub const Json = std.json.Parsed(@This());

    pub const Sampler = struct {
        filter: enum {
            nearest,
            linear,
            mipmap, // default ctype: volume, cubemap, texture

            pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
                return try JsonHelpers.parseEnumAsString(@This(), alloc, source, options);
            }
        },
        wrap: enum {
            clamp,
            repeat, // default ctype: texture, volume

            pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
                return try JsonHelpers.parseEnumAsString(@This(), alloc, source, options);
            }
        },
        vflip: JsonHelpers.StringBool, // default ctype: cubemap is false
        srgb: JsonHelpers.StringBool, // default all false
        internal: enum {
            byte,

            pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
                return try JsonHelpers.parseEnumAsString(@This(), alloc, source, options);
            }
        },
    };
    pub const Ctype = enum {
        texture,
        cubemap,
        buffer,
        volume,
        mic,
        webcam,
        video,
        music,
        keyboard,

        pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            return try JsonHelpers.parseEnumAsString(@This(), alloc, source, options);
        }
    };
    pub const Input = struct {
        id: u32,
        src: []const u8,
        channel: u2, // 0..=3
        // published: u32, // 1?
        sampler: Sampler,
        ctype: Ctype,
    };
    pub const PassType = enum {
        // shared by all shaders
        common,

        // - [Shadertoy Tutorial](https://inspirnathan.com/posts/62-shadertoy-tutorial-part-15)
        // if buffer A uses buffer A as input, last frame's buffer A goes in
        buffer,

        cubemap,

        // this is displayed on screen
        image,

        sound,

        pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            return try JsonHelpers.parseEnumAsString(@This(), alloc, source, options);
        }
    };
    pub const Output = struct {
        id: u32,

        // idk what this is. always seems to be 0??
        channel: u32,
    };
    pub const Pass = struct {
        inputs: []Input,
        outputs: []Output,
        code: []const u8,
        name: []const u8,
        description: []const u8,

        // passes in order: buffer, buffer, buffer, buffer, cubemap, image, sound
        type: PassType,
    };
    pub const ShaderInfo = struct {
        id: []const u8,
        name: []const u8,
        username: []const u8,
        description: []const u8,
    };
    pub const Shader = struct {
        info: ShaderInfo,

        // passes execute in the order they are defined. outputs from one pass may go
        // as inputs to next pass if defined
        renderpass: []Pass,
    };
    Shader: Shader,

    const IapiShader = struct {
        info: ShaderInfo,
        renderpass: []struct {
            inputs: []struct {
                id: []const u8,
                filepath: []const u8,
                previewfilepath: []const u8,
                channel: u2,
                sampler: Sampler,
                type: Ctype,
            },
            outputs: []struct {
                id: []const u8,
                channel: u32,
            },
            code: []const u8,
            name: []const u8,
            description: []const u8,
            type: PassType,
        },

        const IdGen = struct {
            hm: std.StringHashMap(u32),
            id_gen: u32 = 1000,

            fn get_id(self: *@This(), id: []const u8) !u32 {
                if (id_from_string(id)) |known| {
                    return known;
                }
                const res = try self.hm.getOrPut(id);
                if (!res.found_existing) {
                    defer self.id_gen += 1;
                    res.value_ptr.* = self.id_gen;
                }
                return res.value_ptr.*;
            }

            fn id_from_string(id: []const u8) ?u32 {
                if (std.mem.eql(u8, id, "4dXGR8")) {
                    return 257;
                }
                if (std.mem.eql(u8, id, "XsXGR8")) {
                    return 258;
                }
                if (std.mem.eql(u8, id, "4sXGR8")) {
                    return 259;
                }
                if (std.mem.eql(u8, id, "XdfGR8")) {
                    return 260;
                }
                if (std.mem.eql(u8, id, "4dX3Rr")) {
                    return 41;
                }
                if (std.mem.eql(u8, id, "4dXGRr")) {
                    return 33;
                }

                return null;
            }
        };

        fn toShader(self: @This(), arena: std.mem.Allocator) !Shader {
            var idgen = IdGen{
                .hm = std.StringHashMap(u32).init(arena),
            };
            var passes = std.ArrayList(Pass).init(arena);
            for (self.renderpass) |pass| {
                var inputs = std.ArrayList(Input).init(arena);
                var outputs = std.ArrayList(Output).init(arena);
                for (pass.inputs) |input| {
                    try inputs.append(.{
                        .id = try idgen.get_id(input.id),
                        .src = input.filepath,
                        .channel = input.channel,
                        .sampler = input.sampler,
                        .ctype = input.type,
                    });
                }
                for (pass.outputs) |output| {
                    try outputs.append(.{
                        .id = try idgen.get_id(output.id),
                        .channel = output.channel,
                    });
                }
                try passes.append(.{
                    .inputs = try inputs.toOwnedSlice(),
                    .outputs = try outputs.toOwnedSlice(),
                    .code = pass.code,
                    .name = pass.name,
                    .description = pass.description,
                    .type = pass.type,
                });
            }
            return .{
                .info = self.info,
                .renderpass = try passes.toOwnedSlice(),
            };
        }
    };

    pub fn from_json(json: []const u8) !Json {
        const toy = std.json.parseFromSlice(@This(), allocator, json, json_parse_ops) catch {
            const err = std.json.parseFromSlice(struct { Error: []const u8 }, allocator, json, json_parse_ops) catch {
                const toylist = std.json.parseFromSlice([]IapiShader, allocator, json, json_parse_ops) catch |e| {
                    return e;
                };
                errdefer toylist.deinit();
                if (toylist.value.len == 0) {
                    return error.ShaderNotFound;
                }
                return .{
                    .arena = toylist.arena,
                    .value = Toy{
                        .Shader = try toylist.value[0].toShader(toylist.arena.allocator()),
                    },
                };
            };
            defer err.deinit();
            std.debug.print("shadertoy error: {s}\n", .{err.value.Error});
            return error.ShaderNotAccessible;
        };
        return toy;
    }

    pub fn from_file(path: []const u8) !Json {
        const rpath = try std.fs.realpathAlloc(allocator, path);
        defer allocator.free(rpath);
        const file = try std.fs.openFileAbsolute(rpath, .{});
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 10 * 1000 * 1000);
        defer allocator.free(bytes);
        return try from_json(bytes);
    }
};

pub const Cached = struct {
    cache_dir: []const u8,

    fn init(cache_dir: []const u8) !@This() {
        const cache = try std.fs.realpathAlloc(allocator, cache_dir);
        var res = try std.fs.cwd().openDir(cache, .{});
        defer res.close();
        return .{
            .cache_dir = cache,
        };
    }
    fn deinit(self: *@This()) void {
        allocator.free(self.cache_dir);
    }
    fn shader(self: *@This(), id: []const u8) !Toy.Json {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{
            self.cache_dir,
            id,
        });
        defer allocator.free(dir);
        const path = try std.fs.path.join(allocator, &[_][]const u8{
            dir,
            "toy.json",
        });
        defer allocator.free(path);

        const cwd = std.fs.cwd();
        var file = cwd.openFile(path, .{}) catch {
            const json = try get_inner_shader_json(id);
            defer allocator.free(json);
            try cwd.makePath(dir);
            const file = try cwd.createFile(path, .{});
            defer file.close();

            const value = try std.json.parseFromSlice(
                std.json.Value,
                allocator,
                json,
                json_parse_ops,
            );
            defer value.deinit();

            try std.json.stringify(
                &value.value,
                .{ .whitespace = .indent_4 },
                file.writer(),
            );

            return try Toy.from_json(json);
        };
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 10 * 1000 * 1000);
        defer allocator.free(bytes);
        return try Toy.from_json(bytes);
    }

    fn media_bytes(self: *@This(), media_path: []const u8) ![]u8 {
        const path = try std.fs.path.join(allocator, &[_][]const u8{
            self.cache_dir,
            media_path[1..],
        });
        defer allocator.free(path);

        const cwd = std.fs.cwd();
        var file = cwd.openFile(path, .{}) catch {
            const bytes = try get_media_bytes(media_path);
            errdefer allocator.free(bytes);

            try cwd.makePath(std.fs.path.dirname(path).?);
            const file = try cwd.createFile(path, .{});
            defer file.close();

            try file.writeAll(bytes);
            return bytes;
        };
        defer file.close();

        const bytes = try file.readToEndAlloc(allocator, 10 * 1000 * 1000);
        return bytes;
    }

    fn media_img(self: *@This(), media_path: []const u8) !utils.ImageMagick.FloatImage {
        const bytes = try self.media_bytes(media_path);
        defer allocator.free(bytes);

        const img = try utils.ImageMagick.decode_jpg(bytes, .float);
        return img;
    }
};

pub fn get_shader_json(id: []const u8) ![]const u8 {
    const url = try std.fmt.allocPrintZ(allocator, api_base ++ "/shaders/{s}?key=" ++ key, .{id});
    defer allocator.free(url);
    const body = try Curl.get(url);
    return body;
}

pub fn get_inner_shader_json(id: []const u8) ![]const u8 {
    const referer = try std.fmt.allocPrintZ(allocator, "Referer: https://www.shadertoy.com/view/{s}", .{id});
    defer allocator.free(referer);
    const req = try std.fmt.allocPrintZ(allocator, "s=%7B%20%22shaders%22%20%3A%20%5B%22{s}%22%5D%20%7D&nt=1&nl=1&np=1", .{id});
    defer allocator.free(req);

    const body = try Curl.post("https://www.shadertoy.com/shadertoy", &[_][:0]const u8{
        referer,
        "Content-Type: application/x-www-form-urlencoded",
    }, req);
    return body;
}

pub fn get_shader(id: []const u8) !Toy.Json {
    const body = try get_shader_json(id);
    defer allocator.free(body);

    const toy = try Toy.from_json(body);
    return toy;
}

pub fn query_shaders(query: []const u8) !void {
    const encoded = try encode(query);
    defer allocator.free(encoded);

    // sort by: name, love, popular (default), newest, hot
    // const url = try std.fmt.allocPrint(allocator, api_base ++ "/shaders/query/{s}?sort=newest&from=0&num=25&key=" ++ key, .{encoded});
    // defer allocator.free(url);
}

pub fn get_media_bytes(media_path: []const u8) ![]u8 {
    const url = try std.fmt.allocPrintZ(allocator, base ++ "{s}", .{media_path});
    defer allocator.free(url);
    const body = try Curl.get(url);
    return body;
}

fn get(url: []const u8) ![]const u8 {
    std.debug.print("{s}\n", .{url});
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();
    const uri = try std.Uri.parse(url);
    const buf = try allocator.alloc(u8, 1024 * 8);
    defer allocator.free(buf);
    var req = try client.open(.GET, uri, .{
        .server_header_buffer = buf,
        .headers = .{
            .user_agent = .{
                // hello website. i am actuall firefox. no cap.
                .override = "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
            },
        },
    });
    defer req.deinit();
    try req.send();
    // try req.finish();
    // try req.write();
    try req.wait();

    var body = std.ArrayList(u8).init(allocator);
    try req.reader().readAllArrayList(&body, 1000 * 1000);
    return try body.toOwnedSlice();
}

fn encode(str: []const u8) ![]const u8 {
    const al = std.ArrayList(u8).init(allocator);
    std.Uri.Component.percentEncode(al.writer(), str, struct {
        fn isValidChar(c: u8) bool {
            return switch (c) {
                0, '%', ':' => false,
                else => true,
            };
        }
    }.isValidChar);
    return try al.toOwnedSlice();
}
