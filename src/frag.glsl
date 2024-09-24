#version 450

layout(set = 0, binding = 1)
uniform Time {
    float iTime;
};

layout(set = 0, binding = 2)
uniform Resolution {
    vec3 iResolution;
};

layout(location = 0) out vec4 outputColor;
void main() {
    outputColor = vec4(1.0, 0.0, mod(iTime/10.0, 1.0), 1.0);
}
