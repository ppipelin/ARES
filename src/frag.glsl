#version 330
#define PI 3.14159265359

in vec3 var_Wposition;
in vec3 var_Wnormal;
in vec2 var_uv;
in vec2 viewportCoord;

layout (location = 0) out vec4 out_fragColor;

uniform mat4 V;

uniform vec3 uni_WlightDirection; 
uniform vec3 uni_lightColor; 
// 0 -> fresnel / 1 -> phong
uniform uint uni_mode;

uniform vec3 uni_diffuse; 
uniform vec4 uni_glossy;
uniform vec3 uni_ambiant;
uniform sampler2D mean_texture_sampler;
uniform ivec2 uni_resolution;

vec2 to_sphere(vec3 xyz){
    float theta = acos( xyz.z );
    float phi = atan( xyz.y, xyz.x );
    phi += ( phi < 0 ) ? 2*PI : 0; // only if you want [0,2pi)
    return vec2(theta, phi);
}

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

void reflection_shading()
{
    mat4 invV = inverse(V);
    vec3 WcamPos = invV[3].xyz / invV[3].w;
    vec3 WtoView = normalize(WcamPos - var_Wposition);
    vec3 wNormal = normalize(var_Wnormal);
    vec3 WRtoView = reflect(WtoView, wNormal);
    vec2 color = to_sphere(WRtoView) / vec2(PI, 2*PI);
    out_fragColor = vec4(color, 0 , 1);
    
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

    mat4 invV = inverse(V);
    vec3 WcamPos = invV[3].xyz / invV[3].w;
    vec3 WtoView = normalize(WcamPos - var_Wposition);
    

    vec3 wNormal = normalize(var_Wnormal);
    
    
    vec3 Wto_light = -uni_WlightDirection;

    vec3 diffuse = uni_diffuse * max(0, dot(wNormal, Wto_light)) * uni_lightColor;

    vec3 glossy = uni_glossy.xyz;
    float shininess = uni_glossy.w;
    vec3 WRtoView = reflect(WtoView, wNormal);
    
    vec3 specular_direct = glossy * uni_lightColor * pow(max(0, dot(WRtoView, -Wto_light)), shininess);
    

    vec2 spherical = to_sphere(WRtoView);
    
    spherical = spherical * uni_resolution / vec2(PI, 2*PI);
    
    vec3 specular_indirect = glossy * texelFetch(mean_texture_sampler, ivec2(spherical), 0 ).rgb;

    float diffuse_weight = 0.5;
    float specular_direct_weight = 0.3;
    float specular_indirect_weight = 0.5;


    res += diffuse_weight * diffuse + 
        specular_direct_weight * specular_direct +
        specular_indirect_weight * specular_indirect;

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
        case 2u:
            uv_shading();
        break;
        default:
            reflection_shading();
        break;
    }
}
