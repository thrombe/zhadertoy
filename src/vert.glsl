#version 450

float iTime = 0.0f;
vec3 iResolution = vec3(0.0f);

layout(location = 0) out vec2 fragCoord;
// layout(location = 0) in vec3 pos;

void main() {
    vec3 positions[6] = vec3[6](
        vec3(1.0, 1.0, 0.0),
        vec3(-1.0, 1.0, 0.0),
        vec3(1.0, -1.0, 0.0),
        vec3(1.0, -1.0, 0.0),
        vec3(-1.0, 1.0, 0.0),
        vec3(-1.0, -1.0, 0.0)
    );

    vec3 pos = positions[gl_VertexIndex];

    gl_Position = vec4(pos, 1.0);
    fragCoord = gl_Position.xy;
    // fragCoord = pos.xy;
}

