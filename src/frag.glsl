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

layout(set = 0, binding = 0)
uniform Time {
    float iTime;
};

layout(set = 0, binding = 1)
uniform Resolution {
    vec3 iResolution;
};
vec3 iMouse;

// - [[glsl-in] Consider supporting combined image/samplers · Issue #4342 · gfx-rs/wgpu · GitHub](https://github.com/gfx-rs/wgpu/issues/4342)
// - [GLSL: Add option to separate combined image samplers when emitting GLSL · Issue #2236 · KhronosGroup/SPIRV-Cross · GitHub](https://github.com/KhronosGroup/SPIRV-Cross/issues/2236)

// #ifdef ZHADER_CHANNEL0
layout(set = 0, binding = 2)
uniform texture2D iCChannel0;
layout(set = 0, binding = 3)
uniform sampler iSChannel0;
#define iChannel0 sampler2D(iCChannel0, iSChannel0)
// #endif // ZHADER_CHANNEL0

// NOTE: 'sampler' is reserved in vulkan glsl
#define sampler zhader_sampler
 
#ifdef ZHADER_COMMON
#include "common.glsl"
#endif // ZHADER_COMMON

#ifdef ZHADER_INCLUDE_IMAGE
#include "image.glsl"
#endif // ZHADER_INCLUDE_IMAGE

#ifdef ZHADER_INCLUDE_BUF1
#include "buffer1.glsl"
#endif // ZHADER_INCLUDE_BUF1

#ifdef ZHADER_INCLUDE_BUF2
#include "buffer2.glsl"
#endif // ZHADER_INCLUDE_BUF2

#ifdef ZHADER_INCLUDE_BUF3
#include "buffer3.glsl"
#endif // ZHADER_INCLUDE_BUF3

#ifdef ZHADER_INCLUDE_BUF4
#include "buffer4.glsl"
#endif // ZHADER_INCLUDE_BUF4

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;
void main() {
    vec2 res = iResolution.xy;
    vec2 pix = vec2(fragCoord.xy / 2.0 + 0.5) * res;
    // vec2 pix = iResolution.xy * fragCoord;
    // vec2 pix = gl_FragCoord.xy;
    mainImage(fragColor, pix);
}
