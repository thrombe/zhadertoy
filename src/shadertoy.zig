const std = @import("std");

const utils = @import("utils.zig");
const JsonHelpers = utils.JsonHelpers;
const FsFuse = utils.FsFuse;
const Fuse = utils.Fuse;
const Curl = utils.Curl;

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
    // path where the toy should be simlinked
    playground: []const u8,
    shader_fuse: FsFuse,
    shader_cache: Cached,
    active_toy: ActiveToy,

    pub const UpdateEvent = enum {
        Buffer1,
        Buffer2,
        Buffer3,
        Buffer4,
        Image,
        All,
    };

    const vert_glsl = @embedFile("./vert.glsl");
    const frag_glsl = @embedFile("./frag.glsl");

    pub fn init() !@This() {
        const cwd_real = try std.fs.cwd().realpathAlloc(allocator, "./");
        defer allocator.free(cwd_real);
        const playground = try std.fs.path.join(allocator, &[_][]const u8{ cwd_real, "shaders" });

        const fuse = try FsFuse.init(config.paths.playground);
        // TODO: fuse() + default self.active_toy won't work cuz of things like has_common
        try fuse.ctx.channel.send(.All);
        const cache = try Cached.init(config.paths.shadertoys);

        return .{
            .playground = playground,
            .shader_fuse = fuse,
            .shader_cache = cache,
            .active_toy = .{},
        };
    }

    pub fn deinit(self: *@This()) void {
        self.shader_fuse.deinit();
        self.shader_cache.deinit();
        self.active_toy.deinit();
        allocator.free(self.playground);
    }

    pub fn has_updates(self: *@This()) bool {
        return self.shader_fuse.can_recv();
    }

    pub fn try_get_update(self: *@This()) ?UpdateEvent {
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

    pub fn load_shadertoy(self: *@This(), id: []const u8) !void {
        var toy = try self.shader_cache.shader(id);
        defer toy.deinit();

        const path = try std.fs.path.join(allocator, &[_][]const u8{ config.paths.shadertoys, id });
        defer allocator.free(path);
        const target = try std.fs.cwd().realpathAlloc(allocator, path);
        defer allocator.free(target);

        const active = try prepare_toy(&self.shader_cache, self.playground, &toy.value, target);
        self.active_toy.deinit();
        self.active_toy = active;

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

        const active = try prepare_toy(&self.shader_cache, self.playground, &toy.value, target);
        self.active_toy.deinit();
        self.active_toy = active;

        try self.shader_fuse.restart(config.paths.playground);
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
    pub fn prepare_toy(cache: *Cached, playground: []const u8, toy: *Toy, dir_path: []const u8) !ActiveToy {
        var dir = try std.fs.openDirAbsolute(dir_path, .{ .no_follow = true });
        defer dir.close();

        // testing if frag exists to figure out if we need to prep or load the toy
        const prep = blk: {
            dir.access("frag.glsl", .{}) catch break :blk true;
            break :blk false;
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
                    const out_buf = try std.meta.intToEnum(ActiveToy.Buf, out_id);

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

                    buffer.* = .{
                        .output = out_buf,
                        .inputs = try ActiveToy.Inputs.from(pass.inputs, cache),
                    };
                },
                .image => {
                    if (prep) {
                        var buf = try dir.createFile("image.glsl", .{});
                        defer buf.close();

                        try buf.writeAll(pass.code);
                    }

                    t.passes.image.inputs = try ActiveToy.Inputs.from(pass.inputs, cache);
                },
                else => {
                    std.debug.print("unimplemented renderpass type: {any}\n", .{pass.type});
                },
            }
        }

        if (prep) {
            var vert = try dir.createFile("vert.glsl", .{});
            defer vert.close();
            try vert.writeAll(vert_glsl);

            var frag = try dir.createFile("frag.glsl", .{});
            defer frag.close();
            try frag.writeAll(frag_glsl);
        }

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

        t.mark_current_frame_inputs();
        return t;
    }
};

// what do i need to know? (for now, only consider buffers)
//  - pass
//    - output: buffer{1, 2, 3, 4}
//    - 4 inputs: buffer{1, 2, 3, 4}
pub const ActiveToy = struct {
    pub const Buf = enum(u32) {
        BufferA = 257,
        BufferB = 258,
        BufferC = 259,
        BufferD = 260,
    };
    pub const Buffer = union(enum) {
        Buf: Buf,
        keyboard,
        music,
        texture,
    };
    pub const Sampler = Toy.Sampler;
    pub const Input = struct {
        typ: Buffer,
        sampler: Sampler,
        from_current_frame: bool = false,

        fn from(input: Toy.Input, cache: *Cached) !@This() {
            switch (input.ctype) {
                .buffer => {
                    return .{
                        .typ = .{ .Buf = try std.meta.intToEnum(Buf, input.id) },
                        .sampler = input.sampler,
                    };
                },
                .texture => {
                    var img = try cache.media_img(input.src);
                    img.deinit();
                    return .{ .typ = .texture, .sampler = input.sampler };
                },
                .keyboard => {
                    return .{ .typ = .keyboard, .sampler = input.sampler };
                },
                .music => {
                    return .{ .typ = .music, .sampler = input.sampler };
                },
                else => |typ| {
                    std.debug.print("did not handle ctype '{any}'\n", .{typ});
                    @panic("");
                },
            }
        }

        fn mark_current_frame_input(self: *@This(), typ: Buffer) void {
            if (std.meta.eql(self.typ, typ)) {
                self.from_current_frame = true;
            }
        }
    };
    pub const Inputs = struct {
        input1: ?Input = null,
        input2: ?Input = null,
        input3: ?Input = null,
        input4: ?Input = null,

        fn from(inputs: []Toy.Input, cache: *Cached) !@This() {
            var self: @This() = .{};
            for (inputs, 1..) |input, i| {
                switch (i) {
                    1 => {
                        self.input1 = try Input.from(input, cache);
                    },
                    2 => {
                        self.input2 = try Input.from(input, cache);
                    },
                    3 => {
                        self.input3 = try Input.from(input, cache);
                    },
                    4 => {
                        self.input4 = try Input.from(input, cache);
                    },
                    else => return error.TooManyInputs,
                }
            }
            return self;
        }

        fn mark_current_frame_input(self: *@This(), typ: Buffer) void {
            if (self.input1) |*inp| inp.mark_current_frame_input(typ);
            if (self.input2) |*inp| inp.mark_current_frame_input(typ);
            if (self.input3) |*inp| inp.mark_current_frame_input(typ);
            if (self.input4) |*inp| inp.mark_current_frame_input(typ);
        }

        fn mark_current_frame_inputs(self: *@This(), inputs: *@This()) void {
            if (inputs.input1) |inp| {
                self.mark_current_frame_input(inp.typ);
            }
            if (inputs.input2) |inp| {
                self.mark_current_frame_input(inp.typ);
            }
            if (inputs.input3) |inp| {
                self.mark_current_frame_input(inp.typ);
            }
            if (inputs.input4) |inp| {
                self.mark_current_frame_input(inp.typ);
            }
        }
    };
    pub const BufferPass = struct {
        output: Buf,
        inputs: Inputs,

        fn deinit(self: *@This()) void {
            _ = self;
        }
    };

    has_common: bool = false,
    passes: struct {
        image: struct {
            inputs: Inputs = .{},
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
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            if (self.passes.buffer3) |*buf1| {
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            self.passes.image.inputs.mark_current_frame_inputs(&buf.inputs);
        }
        if (self.passes.buffer2) |*buf| {
            if (self.passes.buffer3) |*buf1| {
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            self.passes.image.inputs.mark_current_frame_inputs(&buf.inputs);
        }
        if (self.passes.buffer3) |*buf| {
            if (self.passes.buffer4) |*buf1| {
                buf1.inputs.mark_current_frame_inputs(&buf.inputs);
            }
            self.passes.image.inputs.mark_current_frame_inputs(&buf.inputs);
        }
        if (self.passes.buffer4) |*buf| {
            self.passes.image.inputs.mark_current_frame_inputs(&buf.inputs);
        }
    }

    pub fn deinit(self: *@This()) void {
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
    pub const Input = struct {
        id: u32,
        src: []const u8,
        channel: u2, // 0..=3
        published: u32, // 1?
        sampler: Sampler,
        ctype: enum {
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
        },
    };
    Shader: struct {
        info: struct {
            id: []const u8,
            name: []const u8,
            username: []const u8,
            description: []const u8,
        },

        // passes execute in the order they are defined. outputs from one pass may go
        // as inputs to next pass if defined
        renderpass: []struct {
            inputs: []Input,
            outputs: []struct {
                id: u32,

                // idk what this is. always seems to be 0??
                channel: u32,
            },
            code: []const u8,
            name: []const u8,
            description: []const u8,

            // passes in order: buffer, buffer, buffer, buffer, cubemap, image, sound
            type: enum {
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
            },
        },
    },

    pub fn from_json(json: []const u8) !Json {
        const toy = std.json.parseFromSlice(@This(), allocator, json, json_parse_ops) catch |e| {
            const err = std.json.parseFromSlice(struct { Error: []const u8 }, allocator, json, json_parse_ops) catch {
                return e;
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
            const json = try get_shader_json(id);
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

    fn media_img(self: *@This(), media_path: []const u8) !utils.ImageMagick.Image {
        const bytes = try self.media_bytes(media_path);
        defer allocator.free(bytes);

        const img = try utils.ImageMagick.decode_jpg(bytes);
        return img;
    }
};

pub fn get_shader_json(id: []const u8) ![]const u8 {
    const url = try std.fmt.allocPrintZ(allocator, api_base ++ "/shaders/{s}?key=" ++ key, .{id});
    defer allocator.free(url);
    const body = try Curl.get(url);
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
