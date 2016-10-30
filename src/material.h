#ifndef _MATERIAL_
#define _MATERIAL_

#include "math/vector3.h"

const int MATERIAL_EMISSIVE = 0;
const int MATERIAL_DIFFUSE = 1;
const int MATERIAL_SPECULAR = 2;
const int MATERIAL_REFRACTIVE = 3;

typedef struct
{
	Vector3 c;
	int type;
	
	float ior;
}
Material;

#ifdef __cplusplus
extern "C" {
#endif

Material  material_emissive  (Vector3 e);
Material  material_diffuse   (Vector3 d);
Material  material_specular  (Vector3 s);
Material  material_refractive(Vector3 r, float ior);

#ifdef __cplusplus
}
#endif

#endif