#ifndef _INTERSECTION_
#define _INTERSECTION_

#include "math/vector2.h"
#include "math/vector3.h"
#include "material.h"

typedef struct
{
	bool is_intersect;
	float d;
	Vector3 normal;
	Material* m;
    Vec2f uv;
} 
Hit;

#ifdef __cplusplus
extern "C" {
#endif

__device__  void  hit_set               (Hit* hit, float d, Vector3 normal, Material* m, Vec2f uv);
__device__  void  hit_set_no_intersect  (Hit* hit);

#ifdef __cplusplus
}
#endif

#endif