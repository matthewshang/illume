#include "intersection.h"

__device__
Intersection intersection_create(int is_intersect, float d, Vector3 normal)
{
	Intersection inter;
	inter.is_intersect = is_intersect;
	inter.d = d;
	inter.normal = normal;
	return inter;
}

__device__
Intersection intersection_create_no_intersect()
{
	Intersection inter;
	inter.is_intersect = 0;
	inter.d = -1;
	inter.normal = vector3_create(0, 0, 0);
	return inter;
}