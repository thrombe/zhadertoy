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

pub const FsFuse = struct {
    // - [emcrisostomo/fswatch](https://github.com/emcrisostomo/fswatch?tab=readme-ov-file#libfswatch)
    // - [libfswatch/c/libfswatch.h Reference](http://emcrisostomo.github.io/fswatch/doc/1.17.1/libfswatch.html/libfswatch_8h.html#ae465ef0618fb1dc6d8b70dee68359ea6)
    const c = @cImport({
        @cInclude("libfswatch/c/libfswatch.h");
    });

    const Ctx = struct {
        handle: c.FSW_HANDLE,
        trigger: Fuse,
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
        allocator.free(self.ctx.path);
        allocator.destroy(self.ctx);
    }

    pub fn unfuse(self: *@This()) bool {
        return self.ctx.trigger.unfuse();
    }

    pub fn check(self: *@This()) bool {
        return self.ctx.trigger.check();
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
            .trigger = .{},
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

                var valid = false;
                for (events[0..@intCast(num)]) |event| {
                    for (event.flags[0..event.flags_num]) |f| {
                        if (flags & @as(c_int, @intCast(f)) == 0) {
                            continue;
                        }
                        const name = c.fsw_get_event_flag_name(f);
                        std.debug.print("Path: {s}\n", .{event.path});
                        std.debug.print("Event Type: {s}\n", .{std.mem.span(name)});
                        valid = true;
                    }
                }

                if (valid) {
                    ctx.?.trigger.fuse();
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

    pub fn fuse(self: *@This()) void {
        self.fused.store(true, .release);
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
