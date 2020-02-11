#version 330

in vec3 var_Cposition;
in vec3 var_Wposition;
in vec3 var_Cnormal;
in vec2 var_uv;
// La sortie correspondant à la couleur du fragment
layout (location = 0) out vec4 out_fragColor;
// Declaration des uniforms
uniform vec3 uni_lightPosition; // Position de la lumiere en repere camera
uniform vec3 uni_lightColor; // Couleur de la lumière
uniform vec3 uni_diffuseColor; // Couleur diffuse du materiau
// Calcule la couleur du fragment
vec3 color()
{
    vec3 lightDirection = normalize(uni_lightPosition-var_Wposition);
    return max(dot(lightDirection,normalize(var_Cnormal)), 0.0)*uni_lightColor*uni_diffuseColor;
}

vec3 direction_to_color(vec3 dir)
{
    return abs(dir);
}

// Programme principal
void main()
{
    float alpha = 1.0 - abs(dot(var_Cnormal, normalize(var_Cposition)));
    
    
    //out_fragColor = vec4(color(), alpha);
    //out_fragColor = vec4(1, 0, 0, alpha);
    out_fragColor = vec4(direction_to_color(var_Cnormal), alpha);
    //out_fragColor = vec4(var_uv.x, var_uv.y, 0, alpha);
}
