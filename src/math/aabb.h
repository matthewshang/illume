#ifndef _AABB_
#define _AABB_

#include "vector3.h"
#include "constants.h"
#include "ray.h"

typedef struct
{
	Vector3 min;
	Vector3 max;
}
AABB;

#ifdef __cplusplus
extern "C" {
#endif

            AABB   aabb_create         ();
            void   aabb_update         (AABB* aabb, Vector3 p);
            void   aabb_get_vertices   (AABB aabb, Vector3* vertices);
__device__  float  aabb_ray_intersect  (AABB aabb, Ray ray);

#ifdef __cplusplus
}
#endif

#endif