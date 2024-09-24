#version 450

layout(set = 0, binding = 1)
uniform Time {
    float iTime;
};

layout(set = 0, binding = 2)
uniform Resolution {
    vec3 iResolution;
};


layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;
void main() {
    // outputColor = vec4(1.0, 0.0, mod(iTime/10.0, 1.0), 1.0);
    // mainImage(fragColor, iResolution.xy * fragCoord);
    // fragColor = vec4(fragCoord.xyyy * iResolution.xyyy);
    // fragColor = vec4(fragCoord.xyyy / 2.0 + 0.5) * iResolution.xyyy;
    mainImage(fragColor, vec2(fragCoord.xy / 2.0 + 0.5) * iResolution.xy);
    // mainImage(fragColor, gl_FragCoord.xy);
}
