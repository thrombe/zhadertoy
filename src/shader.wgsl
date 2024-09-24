struct StateUniform {
    render_width: u32,
    render_height: u32,
    display_width: u32,
    display_height: u32,
    windowless: u32,

    time: f32,
    cursor_x: f32,
    cursor_y: f32,

    scroll: f32,
    mouse_left: u32,
    mouse_right: u32,
    mouse_middle: u32,
};

@group(0) @binding(0)
var<uniform> state: StateUniform;

@vertex
fn vert_main(
    @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4<f32> {
    var positions = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(-1.0, 1.0, 0.0),
        vec3<f32>(1.0, -1.0, 0.0),
        vec3<f32>(1.0, -1.0, 0.0),
        vec3<f32>(-1.0, 1.0, 0.0),
        vec3<f32>(-1.0, -1.0, 0.0),
    );
    var pos = positions[VertexIndex];
    return vec4<f32>(pos, 1.0);
}

@fragment
fn frag_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let width = f32(state.display_width);
    let height = f32(state.display_height);

    return vec4<f32>(state.cursor_x/width, state.cursor_y/height, state.time/10.0 % 1.0, 0.0);
}
