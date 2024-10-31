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
    vec3 iChannelResolution[4];
    int cubemap_pass;
};

// TODO: sound :}
float iChannelTime[4];
float iSampleRate = 4800.0;

// - [[glsl-in] Consider supporting combined image/samplers · Issue #4342 · gfx-rs/wgpu · GitHub](https://github.com/gfx-rs/wgpu/issues/4342)
// - [GLSL: Add option to separate combined image samplers when emitting GLSL · Issue #2236 · KhronosGroup/SPIRV-Cross · GitHub](https://github.com/KhronosGroup/SPIRV-Cross/issues/2236)
#ifdef ZHADER_INCLUDE_BUF1
    #include "generated_buffer1.glsl"
#endif // ZHADER_INCLUDE_BUF1

#ifdef ZHADER_INCLUDE_BUF2
    #include "generated_buffer2.glsl"
#endif // ZHADER_INCLUDE_BUF2

#ifdef ZHADER_INCLUDE_BUF3
    #include "generated_buffer3.glsl"
#endif // ZHADER_INCLUDE_BUF3

#ifdef ZHADER_INCLUDE_BUF4
    #include "generated_buffer4.glsl"
#endif // ZHADER_INCLUDE_BUF4

#ifdef ZHADER_INCLUDE_CUBEMAP
    #include "generated_cubemap.glsl"
#endif // ZHADER_INCLUDE_CUBEMAP

#ifdef ZHADER_INCLUDE_IMAGE
    #include "generated_image.glsl"
#endif // ZHADER_INCLUDE_IMAGE

#ifdef ZHADER_INCLUDE_SCREEN
    layout(set = 0, binding = 1)
    uniform texture2D screen_tex;
    layout(set = 0, binding = 2)
    uniform sampler screen_sampler;
    #define screen sampler2D(screen_tex, screen_sampler)
#endif // ZHADER_INCLUDE_SCREEN

// NOTE: reserved keywords in vulkan glsl
#define sampler zhader_sampler
#define textureCube zhader_textureCube
#define buffer zhader_buffer
#define packUnorm2x16 zhader_packUnorm2x16
#define packSnorm2x16 zhader_packSnorm2x16
#define packUnorm4x8 zhader_packUnorm4x8
#define packSnorm4x8 zhader_packSnorm4x8
#define unpackUnorm2x16 zhader_unpackUnorm2x16
#define unpackSnorm2x16 zhader_unpackSnorm2x16
#define unpackUnorm4x8 zhader_unpackUnorm4x8
#define unpackSnorm4x8 zhader_unpackSnorm4x8

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

#ifdef ZHADER_INCLUDE_CUBEMAP
    vec3 cubemap_res = vec3(1024.0, 1024.0, 1.0);
    #define iResolution cubemap_res

    #include "cubemap.glsl"

    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 pos = fragCoord/iResolution.xy;
        // pos.y = 1.0 - pos.y;

        vec3 ray_dir;
        switch (cubemap_pass) {
        case 0: ray_dir = vec3(1.0, 0.0, 0.0); break;
        case 1: ray_dir = vec3(-1.0, 0.0, 0.0); break;
        case 2: ray_dir = vec3(0.0, -1.0, 0.0); break;
        case 3: ray_dir = vec3(0.0, 1.0, 0.0); break;
        case 4: ray_dir = vec3(0.0, 0.0, 1.0); break;
        case 5: ray_dir = vec3(0.0, 0.0, -1.0); break;
        default: fragColor = vec4(0.0); return;
        }
        ray_dir = normalize(ray_dir);

        mainCubemap(fragColor, fragCoord, vec3(0, 0, 0), ray_dir);
    }
#endif // ZHADER_INCLUDE_CUBEMAP

#ifdef ZHADER_INCLUDE_IMAGE
    #include "image.glsl"
#endif // ZHADER_INCLUDE_IMAGE

#ifdef ZHADER_INCLUDE_SCREEN
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 pos = fragCoord/iResolution.xy;
        pos.y = 1.0 - pos.y;
        fragColor = texture(screen, pos);
    }
#endif // ZHADER_INCLUDE_SCREEN

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;
void main() {
    // [-1, 1] -> [0, 1]
    // vec2 pix = vec2(fragCoord.xy / 2.0 + 0.5) * iResolution.xy;
    vec2 pix = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);

    mainImage(fragColor, pix);
}
