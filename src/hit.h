#ifndef _INTERSECTION_
#define _INTERSECTION_

#include "math/vector3.h"
#include "material.h"

typedef struct
{
	int is_intersect;
	float d;
	Vector3 normal;
	Material m;
} 
Hit;

#ifdef __cplusplus
extern "C" {
#endif

__device__  Hit  hit_create               (float d, Vector3 normal, Material m);
__device__  Hit  hit_create_no_intersect  ();

#ifdef __cplusplus
}
#endif

#endif