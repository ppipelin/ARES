#version 330

in vec3 var_Wposition;
in vec3 var_Wnormal;
in vec2 var_uv;

layout (location = 0) out vec4 out_fragColor;

uniform vec3 uni_WlightDirection; 
uniform vec3 uni_lightColor; 

uniform uint uni_mode;

uniform vec3 uni_diffuse; 
uniform vec3 uni_glossy;
uniform vec3 uni_ambiant;


vec3 direction_to_color(vec3 dir)
{
    return abs(dir);
}

void fresnell_normal()
{
    vec3 Wnormal = normalize(var_Wnormal);
    float alpha = 0.5;//abs(dot(Wnormal, normalize(var_Cposition))); // TODO
    alpha = 1.0 - alpha;
    out_fragColor = vec4(direction_to_color(Wnormal), alpha);
}

void phong()
{
    float alpha = 1;
    vec3 res = vec3(0);
    res += uni_ambiant;

    vec3 wNormal = normalize(var_Wnormal);
    vec3 Wto_light = -uni_WlightDirection;

    res += uni_diffuse * max(0, dot(wNormal, Wto_light));

    out_fragColor = vec4(res, alpha);
}

void main()
{
    if(uni_mode != 0u)
    {
        fresnell_normal();
    }
    else
    {
        phong();
    }
}
