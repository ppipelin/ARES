#version 330

in vec3 var_color;

layout (location = 0) out vec4 out_fragColor;


void main()
{
    out_fragColor = vec4(var_color, 1.0);
    // out_fragColor = vec4(1, 1, 0, 1);
    // out_fragColor = vec4(vec3(gl_FragCoord.z), 1);
}