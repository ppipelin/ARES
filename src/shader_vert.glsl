#version 330
// Définition des attributs
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
// Définition des varying
out vec3 var_position;
out vec3 var_normal;
out vec2 var_uv;
// Définition des uniforms
uniform mat4 uni_mat_view;
uniform mat4 uni_mat_projection;
// Fonction calculant la matrice de transformation des normales
mat3 normalMatrix(in mat4 transform)
{
    mat3 t = mat3(transform);
    return transpose(inverse(t));
}
// Le programme principal
void main()
{
    gl_Position = uni_mat_projection * uni_mat_view * vec4(in_position, 1.0);
    vec4 position = uni_mat_view * vec4(in_position, 1.0);
    var_position = position.xyz / position.www; // Passage coordonnées homogenes -> cartesiennes
    var_normal = normalize(normalMatrix(uni_mat_view)*in_normal);
    var_uv = in_uv;
}
