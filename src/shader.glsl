#version 450

float iTime = 0.0f;
vec3 iResolution = vec3(0.0f);

void vert_main() {}
void frag_main() {}

// layout(location = 0) in vec3 position; // Vertex position
// layout(location = 1) in vec3 color;    // Vertex color

// layout(location = 0) out vec3 fragColor; // Output to the fragment shader

// void main() {
//     gl_Position = vec4(position, 1.0); // Set the position of the vertex
//     fragColor = color;                 // Pass the color to the fragment shader
// }

layout(location = 0) in vec3 fragColor; // Input from the vertex shader
layout(location = 1) out vec4 outputColor; // Final output color of the pixel
void main() {
    outputColor = vec4(fragColor, 1.0); // Set the output color with full opacity
}
