#ifndef _INTERSECTION_
#define _INTERSECTION_

#include "vector3.h"

typedef struct
{
	int is_intersect;
	float d;
	Vector3 normal;
}
Intersection;

__device__  Intersection  intersection_create               (int is_intersect, float d, Vector3 normal);
__device__  Intersection  intersection_create_no_intersect  ();

#endif