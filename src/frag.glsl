#version 450

/*
- [How To - Shadertoy BETA](https://www.shadertoy.com/howto)

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform float iChannelTime[4];
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform vec3 iChannelResolution[4];
uniform samplerXX iChanneli;
*/

struct ZhaderUniforms {
    float time;
    float time_delta;
    int frame;
    int width;
    int height;
    float mouse_x;
    float mouse_y;
    bool mouse_left;
    bool mouse_right;
    bool mouse_middle;
};

layout(set = 0, binding = 0)
uniform Uniforms {
    ZhaderUniforms zhader_uniforms;
};

#define iTime zhader_uniforms.time
#define iTimeDelta zhader_uniforms.time_delta
#define iFrame zhader_uniforms.frame
#define iResolution vec3(float(zhader_uniforms.width), float(zhader_uniforms.height), 0.0)
#define iMouse vec4(zhader_uniforms.mouse_x, zhader_uniforms.mouse_y, 0.0, 0.0)

// TODO: sound :}
float iChannelTime[4];
float iSampleRate = 4800.0;

// - [[glsl-in] Consider supporting combined image/samplers · Issue #4342 · gfx-rs/wgpu · GitHub](https://github.com/gfx-rs/wgpu/issues/4342)
// - [GLSL: Add option to separate combined image samplers when emitting GLSL · Issue #2236 · KhronosGroup/SPIRV-Cross · GitHub](https://github.com/KhronosGroup/SPIRV-Cross/issues/2236)

// #ifdef ZHADER_CHANNEL0
layout(set = 0, binding = 1)
uniform texture2D iCChannel0;
layout(set = 0, binding = 2)
uniform sampler iSChannel0;
#define iChannel0 sampler2D(iCChannel0, iSChannel0)
// #endif // ZHADER_CHANNEL0

// #ifdef ZHADER_CHANNEL1
layout(set = 0, binding = 3)
uniform texture2D iCChannel1;
layout(set = 0, binding = 4)
uniform sampler iSChannel1;
#define iChannel1 sampler2D(iCChannel1, iSChannel1)
// #endif // ZHADER_CHANNEL1

// #ifdef ZHADER_CHANNEL2
layout(set = 0, binding = 5)
uniform texture2D iCChannel2;
layout(set = 0, binding = 6)
uniform sampler iSChannel2;
#define iChannel2 sampler2D(iCChannel2, iSChannel2)
// #endif // ZHADER_CHANNEL2

// #ifdef ZHADER_CHANNEL3
layout(set = 0, binding = 7)
uniform texture2D iCChannel3;
layout(set = 0, binding = 8)
uniform sampler iSChannel3;
#define iChannel3 sampler2D(iCChannel3, iSChannel3)
// #endif // ZHADER_CHANNEL3

// NOTE: 'sampler' is reserved in vulkan glsl
#define sampler zhader_sampler

// NOTE: wgsl has isnan, and it's defined here too - but it won't compile
#define isnan isNan
#define isinf isInf

bool isNan(float x) {
    return x != x;
}
bool isNan(double x) {
    return x != x;
}
bool isNan(vec3 v) {
    return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}
bool isNan(vec2 v) {
    return (v.x != v.x) || (v.y != v.y);
}
bool isNan(vec4 v) {
    return (v.x != v.x) || (v.y != v.y) || (v.z != v.z) || (v.w != v.w);
}

#define ZHADER_INF 1e38
bool isInf(float x) {
    return (x > ZHADER_INF || x < -ZHADER_INF);
}
bool isInf(double x) {
    return (x > ZHADER_INF || x < -ZHADER_INF);
}
 
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

#ifdef ZHADER_VFLIP
    pix.y = res.y - pix.y;
#endif

    // vec2 pix = iResolution.xy * fragCoord;
    // vec2 pix = gl_FragCoord.xy;
    mainImage(fragColor, pix);
}
