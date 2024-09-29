#version 450

/*
- [How To - Shadertoy BETA](https://www.shadertoy.com/howto)

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform float iChannelTime[4];
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform vec3 iChannelResolution[4];
uniform samplerXX iChanneli;
*/

layout(set = 0, binding = 1)
uniform Time {
    float iTime;
};

layout(set = 0, binding = 2)
uniform Resolution {
    vec3 iResolution;
};

#ifdef ZHADER_COMMON
#include "common.glsl"
#endif // ZHADER_COMMON

#include "image.glsl"

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;
void main() {
    vec2 res = iResolution.xy;
    vec2 pix = vec2(fragCoord.xy / 2.0 + 0.5) * res;
    // vec2 pix = iResolution.xy * fragCoord;
    // vec2 pix = gl_FragCoord.xy;
    mainImage(fragColor, pix);
}
