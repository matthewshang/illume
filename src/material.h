#ifndef _MATERIAL_
#define _MATERIAL_

#include "rapidjson/document.h"

#include "math/vector3.h"
#include "medium.h"

const int MATERIAL_EMISSIVE = 0;
const int MATERIAL_DIFFUSE = 1;
const int MATERIAL_SPECULAR = 2;
const int MATERIAL_REFRACTIVE = 3;
const int MATERIAL_COOKTORRANCE = 4;
const int MATERIAL_ROUGHREFRACTIVE = 5;

typedef struct
{
	Vector3 c;
	int type;
	
	float ior;
	float roughness;
	Medium medium;
}
Material;

#ifdef __cplusplus
extern "C" {
#endif

Material  material_from_json     (rapidjson::Value& json, Medium m);
Material  material_emissive      (Vector3 e);
Material  material_diffuse       (Vector3 d);
Material  material_specular      (Vector3 s);
Material  material_refractive    (Vector3 r, float ior, Medium m);
Material  material_cooktorrance  (Vector3 r, float ior, float roughness);
Material  material_roughrefrac   (Vector3 r, float ior, float roughness, Medium m);

#ifdef __cplusplus
}
#endif

#endif