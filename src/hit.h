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
	int a;
} 
Hit;

#ifdef __cplusplus
extern "C" {
#endif

__device__  void  hit_set               (Hit* hit, float d, Vector3 normal, Material m);
__device__  void  hit_set_no_intersect  (Hit* hit);

#ifdef __cplusplus
}
#endif

#endif