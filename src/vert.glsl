#version 330
// in object 
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
// Définition des varying
out vec3 var_Wposition;
out vec3 var_Wnormal;
out vec2 var_uv;
out vec2 viewportCoord;
// Définition des uniforms
uniform mat4 V;
uniform mat4 M;
uniform mat4 P;
// Fonction calculant la matrice de transformation des normales
mat3 normalMatrix(in mat4 transform)
{
    mat3 t = mat3(transform);
    return transpose(inverse(t));
}
// Le programme principal
void main()
{
    mat4 invV = inverse(V);
    vec3 WcamPos = invV[3].xyz / invV[3].w;
    vec3 WtoView = normalize(WcamPos - var_Wposition);
    mat4 MVP = P * V * M;
    gl_Position = MVP * vec4(in_position, 1.0);
    vec3 ndc = gl_Position.xyz / gl_Position.w; //perspective divide/normalize
    viewportCoord = ndc.xy * 0.5 + 0.5;
    vec4 WpositionH = (M * vec4(in_position, 1.0));
    var_Wposition = WpositionH.xyz / WpositionH.w;
    var_Wnormal = normalize(normalMatrix(M)*in_normal);
    var_uv = in_uv;
}
