#version 330 core

layout (triangles) in;

layout (line_strip, max_vertices=2) out;

in vec3 var_Wposition[3];
in vec3 var_Wnormal[3];

out vec3 var_color;
 
uniform mat4 V;
uniform mat4 P;
uniform vec3 uni_WlightDirection;

const float MAGNITUDE = 0.5;

// Fonction calculant la matrice de transformation des normales
mat3 normalMatrix(in mat4 transform)
{
    mat3 t = mat3(transform);
    return transpose(inverse(t));
}

void main()
{
    

    vec3 point = (var_Wposition[0] + var_Wposition[1] + var_Wposition[2]) / 3.0;
    vec3 normal = (var_Wnormal[0] + var_Wnormal[1] + var_Wnormal[2]) / 3.0;
    normal = normalize(normal);

    mat4 invV = inverse(V);
    vec3 WcamPos = invV[3].xyz / invV[3].w;
    vec3 WtoView = normalize(WcamPos - point);
    vec3 Wreflected = reflect(WtoView, normal);

    vec3 direction = normal; 

    vec3 point1 = point + 0.01*direction;
    vec3 point2 = point1 + direction * MAGNITUDE;
    mat4 VP = P * V;
    gl_Position = VP * vec4(point1, 1);
    var_color = vec3(0, 0, 1);
    EmitVertex();
    gl_Position = VP * vec4(point2, 1);
    var_color = vec3(1, 1, 0);
    EmitVertex();
    EndPrimitive();
}