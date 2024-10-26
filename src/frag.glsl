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

layout(set = 0, binding = 0)
uniform Uniforms {
    float iTime;
    float iTimeDelta;
    int iFrame;
    int _pad;
    vec3 iResolution;
    int _pad2;
    vec4 iMouse;

    // int width;
    // int height;
    // float mouse_x;
    // float mouse_y;
    // bool mouse_left;
    // bool mouse_right;
    // bool mouse_middle;
};
layout(set = 0, binding = 1)
uniform ChannelUniforms {
    ivec4 vflips;
};

// TODO: sound :}
float iChannelTime[4];
float iSampleRate = 4800.0;

// - [[glsl-in] Consider supporting combined image/samplers · Issue #4342 · gfx-rs/wgpu · GitHub](https://github.com/gfx-rs/wgpu/issues/4342)
// - [GLSL: Add option to separate combined image samplers when emitting GLSL · Issue #2236 · KhronosGroup/SPIRV-Cross · GitHub](https://github.com/KhronosGroup/SPIRV-Cross/issues/2236)

#define ZHADER_CHANNEL(index, binding1, binding2) \
    layout(set = 0, binding = binding1) \
    uniform texture2D zhader_channel_##index; \
    layout(set = 0, binding = binding2) \
    uniform sampler zhader_sampler_##index; \
    struct ZhaderChannel##index { \
        int a; \
    }; \
    ZhaderChannel##index iChannel##index; \
    vec4 texture(ZhaderChannel##index chan, vec2 pos) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return texture(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos); \
    } \
    vec4 textureProj(ZhaderChannel##index chan, vec4 pos) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return textureProj(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos); \
    } \
    ivec2 textureSize(ZhaderChannel##index chan, int lod) { \
        return textureSize(sampler2D(zhader_channel_##index, zhader_sampler_##index), lod); \
    } \
    vec4 texelFetch(ZhaderChannel##index chan, ivec2 pos, int lod) { \
        if (vflips[index] == 1) { \
            pos.y = textureSize(chan, lod).y - pos.y - 1; \
        } \
        return texelFetch(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos, lod); \
    } \
    vec4 textureLod(ZhaderChannel##index chan, vec2 pos, float lod) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return textureLod(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos, lod); \
    } \
    vec4 textureGrad(ZhaderChannel##index chan, vec2 pos, vec2 dpdx, vec2 dpdy) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return textureGrad(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos, dpdx, dpdy); \
    } \
    vec4 textureGather(ZhaderChannel##index chan, vec2 pos) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return textureGather(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos); \
    } \
    int textureQueryLevels(ZhaderChannel##index chan) { \
        return textureQueryLevels(sampler2D(zhader_channel_##index, zhader_sampler_##index)); \
    } \
    vec2 textureQueryLod(ZhaderChannel##index chan, vec2 pos) { \
        if (vflips[index] == 1) { \
            pos.y = 1.0 - pos.y; \
        } \
        return textureQueryLod(sampler2D(zhader_channel_##index, zhader_sampler_##index), pos); \
    }

#ifdef ZHADER_CHANNEL0
ZHADER_CHANNEL(0, 2, 3)
#endif // ZHADER_CHANNEL0

#ifdef ZHADER_CHANNEL1
ZHADER_CHANNEL(1, 4, 5)
#endif // ZHADER_CHANNEL1

#ifdef ZHADER_CHANNEL2
ZHADER_CHANNEL(2, 6, 7)
#endif // ZHADER_CHANNEL2

#ifdef ZHADER_CHANNEL3
ZHADER_CHANNEL(3, 8, 9)
#endif // ZHADER_CHANNEL3

#undef ZHADER_CHANNEL

#ifdef ZHADER_INCLUDE_SCREEN
layout(set = 0, binding = 1)
uniform texture2D screen_tex;
layout(set = 0, binding = 2)
uniform sampler screen_sampler;
#define screen sampler2D(screen_tex, screen_sampler)
#endif // ZHADER_INCLUDE_SCREEN

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
 
#ifdef ZHADER_INCLUDE_SCREEN
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 pos = fragCoord/iResolution.xy;
    pos.y = 1.0 - pos.y;
    fragColor = texture(screen, pos);
}
#endif // ZHADER_INCLUDE_SCREEN

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
