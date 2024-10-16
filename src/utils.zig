const std = @import("std");
const main = @import("main.zig");
const allocator = main.allocator;

pub const JsonHelpers = struct {
    pub const StringBool = enum {
        true,
        false,

        pub fn jsonParse(alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            switch (try source.nextAlloc(alloc, options.allocate orelse .alloc_if_needed)) {
                .string, .allocated_string => |field| {
                    if (std.mem.eql(u8, field, "true")) {
                        return .true;
                    } else if (std.mem.eql(u8, field, "false")) {
                        return .false;
                    } else {
                        return error.InvalidEnumTag;
                    }
                },
                else => |token| {
                    std.debug.print("{any}\n", .{token});
                    return error.UnexpectedToken;
                },
            }
        }
    };

    pub fn parseUnionAsStringWithUnknown(t: type, alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !t {
        switch (try source.nextAlloc(alloc, options.allocate orelse .alloc_if_needed)) {
            .string, .allocated_string => |field| {
                inline for (@typeInfo(t).Union.fields) |typ| {
                    comptime if (std.meta.stringToEnum(std.meta.Tag(t), typ.name).? == .unknown) {
                        continue;
                    };

                    if (std.mem.eql(u8, field, typ.name)) {
                        return @unionInit(t, typ.name, {});
                    }
                }
                return .{ .unknown = field };
            },
            else => return error.UnexpectedToken,
        }
    }

    pub fn parseEnumAsString(t: type, alloc: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !t {
        @setEvalBranchQuota(2000);
        switch (try source.nextAlloc(alloc, options.allocate orelse .alloc_if_needed)) {
            .string, .allocated_string => |field| {
                inline for (@typeInfo(t).Enum.fields) |typ| {
                    if (std.mem.eql(u8, field, typ.name)) {
                        return comptime std.meta.stringToEnum(t, typ.name).?;
                    }
                }
                std.debug.print("unknown type: {s}\n", .{field});
                return error.InvalidEnumTag;
            },
            else => |token| {
                std.debug.print("{any}\n", .{token});
                return error.UnexpectedToken;
            },
        }
    }
};

// assumes ok has ok.deinit()
pub fn Result(ok: type, err_typ: type) type {
    return union(enum) {
        Ok: ok,
        Err: Error,

        pub const Error = struct {
            err: err_typ,
            msg: []u8,

            // pub fn owned(err: err_typ, msg: []const u8) !@This() {
            //     return .{
            //         .err = err,
            //         .msg = try allocator.dupe(u8, msg),
            //     };
            // }
            // pub fn deinit(self: *@This()) void {
            //     allocator.free(self.msg);
            // }
        };

        // pub fn deinit(self: *@This()) void {
        //     switch (self) {
        //         .Ok => |res| {
        //             if (std.meta.hasMethod(ok, "deinit")) {
        //                 res.deinit();
        //             }
        //         },
        //         .Err => |err| {
        //             err.deinit();
        //         },
        //     }
        // }
    };
}

pub fn Deque(typ: type) type {
    return struct {
        allocator: std.mem.Allocator,
        buffer: []typ,
        size: usize,

        // fill this index next
        front: usize, // at
        back: usize, // one to the right

        pub fn init(alloc: std.mem.Allocator) !@This() {
            const len = 32;
            const buffer = try alloc.alloc(typ, len);
            return .{
                .allocator = alloc,
                .buffer = buffer,
                .front = 0,
                .back = 0,
                .size = 0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.buffer);
        }

        pub fn push_front(self: *@This(), value: typ) !void {
            if (self.size == self.buffer.len) {
                try self.resize();
                return self.push_front(value) catch unreachable;
            }
            self.front = (self.front + self.buffer.len - 1) % self.buffer.len;
            self.buffer[self.front] = value;
            self.size += 1;
        }

        pub fn push_back(self: *@This(), value: typ) !void {
            if (self.size == self.buffer.len) {
                try self.resize();
                return self.push_back(value) catch unreachable;
            }
            self.buffer[self.back] = value;
            self.back = (self.back + 1) % self.buffer.len;
            self.size += 1;
        }

        pub fn pop_front(self: *@This()) ?typ {
            if (self.size == 0) {
                return null;
            }
            const value = self.buffer[self.front];
            self.front = (self.front + 1) % self.buffer.len;
            self.size -= 1;
            return value;
        }

        pub fn pop_back(self: *@This()) ?typ {
            if (self.size == 0) {
                return null;
            }
            self.back = (self.back + self.buffer.len - 1) % self.buffer.len;
            const value = self.buffer[self.back];
            self.size -= 1;
            return value;
        }

        pub fn peek_front(self: *@This()) ?*const typ {
            if (self.size == 0) {
                return null;
            }
            return &self.buffer[self.front];
        }

        pub fn peek_back(self: *@This()) ?*const typ {
            if (self.size == 0) {
                return null;
            }
            const back = (self.back + self.buffer.len - 1) % self.buffer.len;
            return &self.buffer[back];
        }

        pub fn is_empty(self: *@This()) bool {
            return self.size == 0;
        }

        fn resize(self: *@This()) !void {
            std.debug.assert(self.size == self.buffer.len);

            const size = self.buffer.len * 2;
            const buffer = try self.allocator.alloc(typ, size);
            @memcpy(buffer[0 .. self.size - self.front], self.buffer[self.front..]);
            @memcpy(buffer[self.size - self.front .. self.size], self.buffer[0..self.front]);
            const new = @This(){
                .allocator = self.allocator,
                .buffer = buffer,
                .front = 0,
                .back = self.size,
                .size = self.size,
            };
            self.allocator.free(self.buffer);
            self.* = new;
        }
    };
}

pub fn Channel(typ: type) type {
    return struct {
        const Dq = Deque(typ);
        const Pinned = struct {
            dq: Dq,
            lock: std.Thread.Mutex = .{},
        };
        pinned: *Pinned,

        pub fn init(alloc: std.mem.Allocator) !@This() {
            const dq = try Dq.init(alloc);
            const pinned = try alloc.create(Pinned);
            pinned.* = .{
                .dq = dq,
            };
            return .{
                .pinned = pinned,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.pinned.lock.lock();
            // defer self.pinned.lock.unlock();
            self.pinned.dq.deinit();
            self.pinned.dq.allocator.destroy(self.pinned);
        }

        pub fn send(self: *@This(), val: typ) !void {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            try self.pinned.dq.push_back(val);
        }

        pub fn try_recv(self: *@This()) ?typ {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            return self.pinned.dq.pop_front();
        }

        pub fn can_recv(self: *@This()) bool {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            return self.pinned.dq.peek_front() != null;
        }
    };
}

pub const FsFuse = struct {
    // - [emcrisostomo/fswatch](https://github.com/emcrisostomo/fswatch?tab=readme-ov-file#libfswatch)
    // - [libfswatch/c/libfswatch.h Reference](http://emcrisostomo.github.io/fswatch/doc/1.17.1/libfswatch.html/libfswatch_8h.html#ae465ef0618fb1dc6d8b70dee68359ea6)
    const c = @cImport({
        @cInclude("libfswatch/c/libfswatch.h");
    });

    const Event = union(enum) {
        All,
        File: []const u8,
    };
    const Chan = Channel(Event);
    const Ctx = struct {
        handle: c.FSW_HANDLE,
        channel: Chan,
        path: [:0]const u8,
    };

    ctx: *Ctx,
    thread: std.Thread,

    pub fn testfn() !void {
        var watch = try init("./shaders");
        defer watch.deinit();

        std.time.sleep(std.time.ns_per_s * 10);
    }

    pub fn init(path: [:0]const u8) !@This() {
        const rpath = try std.fs.cwd().realpathAlloc(allocator, path);
        defer allocator.free(rpath);
        const pathZ = try allocator.dupeZ(u8, rpath);

        const watch = try start(pathZ);
        return watch;
    }

    pub fn deinit(self: @This()) void {
        _ = c.fsw_stop_monitor(self.ctx.handle);
        _ = c.fsw_destroy_session(self.ctx.handle);
        self.thread.join();
        self.ctx.channel.deinit();
        allocator.free(self.ctx.path);
        allocator.destroy(self.ctx);
    }

    pub fn can_recv(self: *@This()) bool {
        return self.ctx.channel.can_recv();
    }

    pub fn try_recv(self: *@This()) ?Event {
        return self.ctx.channel.try_recv();
    }

    pub fn restart(self: *@This(), path: [:0]const u8) !void {
        self.deinit();
        self.* = try init(path);
    }

    fn start(path: [:0]const u8) !@This() {
        const ok = c.fsw_init_library();
        if (ok != c.FSW_OK) {
            return error.CouldNotCreateFsWatcher;
        }

        const ctxt = try allocator.create(Ctx);
        ctxt.* = .{
            .channel = try Chan.init(allocator),
            .handle = null,
            .path = path,
        };

        const Callbacks = struct {
            fn spawn(ctx: *Ctx) !void {
                ctx.handle = c.fsw_init_session(c.filter_include) orelse return error.CouldNotInitFsWatcher;
                var oke = c.fsw_add_path(ctx.handle, ctx.path.ptr);
                if (oke != c.FSW_OK) {
                    return error.PathAdditionFailed;
                }

                oke = c.fsw_set_recursive(ctx.handle, true);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }
                oke = c.fsw_set_latency(ctx.handle, 1.0);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }
                oke = c.fsw_set_callback(ctx.handle, @ptrCast(&event_callback), ctx);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }

                std.debug.print("starting monitor\n", .{});
                oke = c.fsw_start_monitor(ctx.handle);
                if (oke != c.FSW_OK) {
                    return error.CouldNotStartWatcher;
                }
            }
            fn event_callback(events: [*c]const c.fsw_cevent, num: c_uint, ctx: ?*Ctx) callconv(.C) void {
                var flags = c.NoOp;
                // flags |= c.Created;
                flags |= c.Updated;
                // flags |= c.Removed;
                // flags |= c.Renamed;
                // flags |= c.OwnerModified;
                // flags |= c.AttributeModified;
                // flags |= c.MovedFrom;
                // flags |= c.MovedTo;

                for (events[0..@intCast(num)]) |event| {
                    for (event.flags[0..event.flags_num]) |f| {
                        if (flags & @as(c_int, @intCast(f)) == 0) {
                            continue;
                        }
                        const name = c.fsw_get_event_flag_name(f);
                        std.debug.print("Path: {s}\n", .{event.path});
                        std.debug.print("Event Type: {s}\n", .{std.mem.span(name)});

                        if (ctx) |cctx| {
                            const stripped = std.fs.path.relative(allocator, cctx.path, std.mem.span(event.path)) catch unreachable;
                            cctx.channel.send(.{ .File = stripped }) catch unreachable;
                        } else {
                            std.debug.print("Error: Event ignored!", .{});
                        }
                    }
                }
            }
        };

        const t = try std.Thread.spawn(.{ .allocator = allocator }, Callbacks.spawn, .{ctxt});
        return .{
            .ctx = ctxt,
            .thread = t,
        };
    }
};

pub const Fuse = struct {
    fused: std.atomic.Value(bool) = .{ .raw = false },

    pub fn fuse(self: *@This()) bool {
        return self.fused.swap(true, .release);
    }
    pub fn unfuse(self: *@This()) bool {
        const res = self.fused.swap(false, .release);
        return res;
    }
    pub fn check(self: *@This()) bool {
        return self.fused.load(.acquire);
    }
};

pub const Curl = struct {
    // - [HTTP Get - Zig cookbook](https://cookbook.ziglang.cc/05-01-http-get.html)
    // - [libcurl - API](https://curl.se/libcurl/c/)
    const c = @cImport({
        @cInclude("curl/curl.h");
    });
    const user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0";

    fn write_callback(ptr: [*]const u8, size: usize, nmemb: usize, userdata: *anyopaque) callconv(.C) usize {
        const total_size = size * nmemb;

        const body: *std.ArrayList(u8) = @ptrCast(@alignCast(userdata));
        body.appendSlice(ptr[0..total_size]) catch unreachable;

        return total_size;
    }

    pub fn get(url: [:0]const u8) ![]u8 {
        var ok = c.curl_global_init(c.CURL_GLOBAL_DEFAULT);
        if (ok != c.CURLE_OK) {
            return error.CurlIsNotOkay;
        }
        const curl = c.curl_easy_init() orelse return error.CurlInitFailed;
        defer c.curl_easy_cleanup(curl);

        var body = std.ArrayList(u8).init(allocator);

        const headers = c.curl_slist_append(null, "user-agent: " ++ user_agent);
        defer c.curl_slist_free_all(headers);

        ok = c.curl_easy_setopt(curl, c.CURLOPT_URL, url.ptr);
        if (ok != c.CURLE_OK) {
            return error.CurlSetOptFailed;
        }
        ok = c.curl_easy_setopt(curl, c.CURLOPT_WRITEFUNCTION, write_callback);
        if (ok != c.CURLE_OK) {
            return error.CurlSetOptFailed;
        }
        ok = c.curl_easy_setopt(curl, c.CURLOPT_WRITEDATA, &body);
        if (ok != c.CURLE_OK) {
            return error.CurlSetOptFailed;
        }
        ok = c.curl_easy_setopt(curl, c.CURLOPT_HTTPHEADER, headers);
        if (ok != c.CURLE_OK) {
            return error.CurlSetOptFailed;
        }

        ok = c.curl_easy_perform(curl);
        if (ok != c.CURLE_OK) {
            return error.CurlRequestFailed;
        }

        return body.toOwnedSlice();
    }
};

pub const ImageMagick = struct {
    // - [ImageMagick – Sitemap](https://imagemagick.org/script/sitemap.php#program-interfaces)
    // - [ImageMagick – MagickWand, C API](https://imagemagick.org/script/magick-wand.php)
    // - [ImageMagick – MagickCore, Low-level C API](https://imagemagick.org/script/magick-core.php)
    const magick = @cImport({
        // @cInclude("MagickCore/MagickCore.h");
        @cDefine("MAGICKCORE_HDRI_ENABLE", "1");
        @cInclude("MagickWand/MagickWand.h");
    });

    pub const Pixel = extern struct {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    };
    pub const Image = struct {
        buffer: []Pixel,
        height: usize,
        width: usize,

        pub fn deinit(self: *@This()) void {
            allocator.free(self.buffer);
        }
    };

    pub fn decode_jpg(bytes: []u8) !Image {
        magick.MagickWandGenesis();
        const wand = magick.NewMagickWand() orelse {
            return error.CouldNotGetWand;
        };
        defer _ = magick.DestroyMagickWand(wand);

        // const pwand = magick.NewPixelWand() orelse {
        //     return error.CouldNotGetWand;
        // };
        // defer _ = magick.DestroyPixelWand(pwand);
        // if (magick.PixelSetColor(pwand, "#28282800") == magick.MagickFalse) {
        //     return error.CouldNotSetPWandColor;
        // }
        // if (magick.MagickSetBackgroundColor(wand, pwand) == magick.MagickFalse) {
        //     return error.CouldNotSetBgColor;
        // }

        if (magick.MagickReadImageBlob(wand, bytes.ptr, bytes.len) == magick.MagickFalse) {
            return error.CouldNotReadImage;
        }

        const img_width = magick.MagickGetImageWidth(wand);
        const img_height = magick.MagickGetImageHeight(wand);

        const buffer = try allocator.alloc(Pixel, img_width * img_height);
        errdefer allocator.free(buffer);

        if (magick.MagickExportImagePixels(
            wand,
            0,
            0,
            img_width,
            img_height,
            "RGBA",
            magick.FloatPixel,
            buffer.ptr,
        ) == magick.MagickFalse) {
            return error.CouldNotRenderToBuffer;
        }

        return .{
            .buffer = buffer,
            .height = img_height,
            .width = img_width,
        };
    }
};
