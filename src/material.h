#ifndef _MATERIAL_
#define _MATERIAL_

#include "vector3.h"

typedef struct
{
	Vector3 e;
	Vector3 d;
}
Material;

#ifdef __cplusplus
extern "C" {
#endif

Material  material_emissive  (Vector3 e);
Material  material_diffuse   (Vector3 d);

#ifdef __cplusplus
}
#endif

#endif