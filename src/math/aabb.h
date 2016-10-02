#ifndef _AABB_
#define _AABB_

#include <math.h>

#include "vector3.h"
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

            AABB   aabb_create          ();
            AABB   aabb_from_vertices   (Vector3 min, Vector3 max);
            void   aabb_update          (AABB* aabb, Vector3 p);
            void   aabb_get_vertices    (AABB aabb, Vector3* vertices);
__device__  __host__ void   aabb_ray_get_points  (AABB aabb, Ray ray, float* entry, float* exit);
__device__  float  aabb_ray_exit        (AABB aabb, Ray ray);
__device__  float  aabb_ray_intersect   (AABB aabb, Ray ray);
			int    aabb_aabb_intersect  (AABB u, AABB v);

#ifdef __cplusplus
}
#endif

#endif