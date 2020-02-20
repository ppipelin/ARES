#version 330
// in object 
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
// Définition des varying
out vec3 var_Wposition;
out vec3 var_Wnormal;
out vec2 var_uv;
// Définition des uniforms
uniform mat4 uni_mat_V;
uniform mat4 uni_mat_M;
uniform mat4 uni_mat_P;
// Fonction calculant la matrice de transformation des normales
mat3 normalMatrix(in mat4 transform)
{
    mat3 t = mat3(transform);
    return transpose(inverse(t));
}
// Le programme principal
void main()
{
    mat4 MVP = uni_mat_P * uni_mat_V * uni_mat_M;
    gl_Position = MVP * vec4(in_position, 1.0);
    vec4 WpositionH = (uni_mat_M * vec4(in_position, 1.0));
    var_Wposition = WpositionH.xyz / WpositionH.w;
    var_Wnormal = normalize(normalMatrix(uni_mat_M)*in_normal);
    var_uv = in_uv;
}
