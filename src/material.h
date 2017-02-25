#ifndef _MATERIAL_
#define _MATERIAL_

#include <vector>

#include "rapidjson/document.h"

#include "math/vector3.h"
#include "medium.h"
#include "texture.h"

const int MATERIAL_EMISSIVE        = 0;
const int MATERIAL_DIFFUSE         = 1;
const int MATERIAL_SPECULAR        = 2;
const int MATERIAL_REFRACTIVE      = 3;
const int MATERIAL_ROUGHREFLECTIVE = 4;
const int MATERIAL_ROUGHREFRACTIVE = 5;
const int MATERIAL_CONDUCTOR       = 6;
const int MATERIAL_ROUGHCONDUCTOR  = 7;
const int MATERIAL_PLASTIC         = 8;

struct Material
{
    Texture albedo;
	int type;
	
	float ior;
    float diffuse_fresnel;
	Texture roughness;
	Medium medium;
    Vector3 eta;
	Vector3 k;
};

#ifdef __cplusplus
extern "C" {
#endif

Material  material_from_json     (rapidjson::Value& json, Medium m, std::vector<Texture>& texture_cache);
Material  material_emissive      (Texture e);
Material  material_diffuse       (Texture d);
Material  material_specular      (Texture s);
Material  material_refractive    (Texture r, float ior, Medium m);
Material  material_roughreflec   (Texture r, float ior, Texture roughness);
Material  material_roughrefrac   (Texture r, float ior, Texture roughness, Medium m);
Material  material_conductor     (Vector3 eta, Vector3 k);
Material  material_roughconductor(Vector3 eta, Vector3 k, Texture roughness);
Material  material_plastic       (Texture specular, float ior);

#ifdef __cplusplus
}
#endif

#endif