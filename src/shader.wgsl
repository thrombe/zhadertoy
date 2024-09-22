// struct Uniform {
//     pos: vec4<f32>,
// 	scale: f32,
// };

// @group(0) @binding(0) var<uniform> in : Uniform;

@vertex fn vertex_main(
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

@fragment fn frag_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}
