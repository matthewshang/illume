#ifndef _PLANE_
#define _PLANE_

#include "vector3.h"
#include "material.h"
#include "ray.h"
#include "intersection.h"

typedef struct
{
	Vector3 p;
	Vector3 n;
	Material m;
}
Plane;

#ifdef __cplusplus
extern "C" {
#endif

            Plane         plane_create         (Vector3 p, Vector3 n, Material m);
__device__  Intersection  plane_ray_intersect  (Plane p, Ray r);

#ifdef __cplusplus
}
#endif

#endif
