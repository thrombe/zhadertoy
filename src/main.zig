const std = @import("std");

const mach = @import("mach");

// using wgpu instead of mach.gpu for better lsp results
const gpu = mach.wgpu;

// The global list of Mach modules registered for use in our application.
pub const modules = .{
    mach.Core,
    // @import("App.zig"),
    // @import("Renderer.zig"),
    App,
    Renderer,
};

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
        windowless: u32,
        time: f32,
        cursor_x: f32,
        cursor_y: f32,
        scroll: f32,

        // TODO: figure out how to send bool or compress this into a single variable
        // can shove inside a u32 and do (variable & u32(<2^n>)) to get it out
        mouse_left: u32,
        mouse_right: u32,
        mouse_middle: u32,
    };
    // uniform bind group offset must be 256-byte aligned
    const uniform_offset = 256;

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

        const shader_module = device.createShaderModuleWGSL("shader.wgsl", @embedFile("shader.wgsl"));
        defer shader_module.release();

        const blend = gpu.BlendState{};
        const color_target = gpu.ColorTargetState{
            .format = core_mod.get(core.main_window, .framebuffer_format).?,
            .blend = &blend,
            .write_mask = gpu.ColorWriteMaskFlags.all,
        };
        const fragment = gpu.FragmentState.init(.{
            .module = shader_module,
            .entry_point = "frag_main",
            .targets = &.{color_target},
        });
        const vertex = gpu.VertexState{
            .module = shader_module,
            .entry_point = "vertex_main",
        };

        const label = @tagName(name) ++ ".init";
        const uniform_buffer = device.createBuffer(&.{
            .label = label ++ " uniform buffer",
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(StateUniform) * uniform_offset, // HUH: why is this size*uniform_offset? it should be the next multiple of uniform_offset
            .mapped_at_creation = .false,
        });

        const bind_group_layout_entry = gpu.BindGroupLayout.Entry.buffer(0, .{ .vertex = true }, .uniform, true, 0);
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
            .bind_group = bind_group,
            .pipeline = pipeline,
            .uniform_buffer = uniform_buffer,
            .state = std.mem.zeroes(StateUniform),
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

        // Present the frame
        core_mod.schedule(.present_frame);
    }

    fn tick(self_mod: *Mod) !void {
        self_mod.schedule(.render_frame);

        var iter = mach.core.pollEvents();
        while (iter.next()) |event| {
            switch (event) {
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

        renderer_mod.schedule(.init);

        self_mod.init(.{});
    }

    fn deinit(core_mod: *mach.Core.Mod, renderer_mod: *Renderer.Mod) !void {
        renderer_mod.schedule(.deinit);
        core_mod.schedule(.deinit);
    }

    fn tick(
        renderer_mod: *Renderer.Mod,
        core_mod: *mach.Core.Mod,
    ) !void {
        renderer_mod.schedule(.tick);

        var iter = mach.core.pollEvents();

        while (iter.next()) |event| {
            switch (event) {
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
                .close => core_mod.schedule(.exit),
                else => {},
            }
        }
    }
};

// TODO: move this to a mach "entrypoint" zig module
pub fn main() !void {
    // Initialize mach core
    try mach.core.initModule();

    // Main loop
    while (try mach.core.tick()) {}
}
