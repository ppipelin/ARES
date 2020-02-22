#version 330

in vec3 var_Wposition;
in vec3 var_Wnormal;
in vec2 var_uv;

layout (location = 0) out vec4 out_fragColor;

uniform mat4 V;

uniform vec3 uni_WlightDirection; 
uniform vec3 uni_lightColor; 
// 0 -> fresnel / 1 -> phong
uniform uint uni_mode;

uniform vec3 uni_diffuse; 
uniform vec4 uni_glossy;
uniform vec3 uni_ambiant;


vec3 direction_to_color(vec3 dir)
{
    return abs(dir);
}

void normal_shading()
{
    vec3 Wnormal = normalize(var_Wnormal);
    float alpha = 0.5;//abs(dot(Wnormal, normalize(var_Cposition))); // TODO
    alpha = 1.0 - alpha;
    out_fragColor = vec4(direction_to_color(Wnormal), alpha);
}

void uv_shading()
{
    vec3 Wnormal = normalize(var_Wnormal);
    float alpha = 0.5;//abs(dot(Wnormal, normalize(var_Cposition))); // TODO
    alpha = 1.0 - alpha;
    out_fragColor = vec4(var_uv.x, 0.1, var_uv.y, alpha);
}

void phong()
{
    float alpha = 1;
    vec3 res = vec3(0);
    res += uni_ambiant;

    vec3 WcamPos = -V[3].xyz / V[3].w;
    vec3 WtoView = normalize(WcamPos - var_Wposition);
    

    vec3 wNormal = normalize(var_Wnormal);
    //if(dot(wNormal, WtoView) < 0)   wNormal = -wNormal;
    vec3 Wto_light = -uni_WlightDirection;

    res += uni_diffuse * max(0, dot(wNormal, Wto_light)) * uni_lightColor;

    vec3 glossy = uni_glossy.xyz;
    float shininess = uni_glossy.w;
    vec3 WRtoView = reflect(WtoView, wNormal);
    
    res += glossy * uni_lightColor * pow(max(0, dot(WRtoView, Wto_light)), shininess);
    out_fragColor = vec4(res, alpha);
}

void main()
{
    switch(uni_mode)
    {
        case 0u:
            phong();
        break;
        case 1u:
            normal_shading();
        break;
        default:
            uv_shading();
        break;
    }
}
