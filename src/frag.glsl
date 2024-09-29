#version 450

layout(set = 0, binding = 1)
uniform Time {
    float iTime;
};

layout(set = 0, binding = 2)
uniform Resolution {
    vec3 iResolution;
};

#include "./image.glsl"

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;
void main() {
    vec2 res = iResolution.xy;
    vec2 pix = vec2(fragCoord.xy / 2.0 + 0.5) * res;
    // vec2 pix = iResolution.xy * fragCoord;
    // vec2 pix = gl_FragCoord.xy;
    mainImage(fragColor, pix);
}
