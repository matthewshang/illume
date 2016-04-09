#include "intersection.h"

__device__
Intersection intersection_create(float d, Vector3 normal, Material m)
{
	Intersection inter;
	inter.is_intersect = 1;
	inter.d = d;
	inter.normal = normal;
	inter.m = m;
	return inter;
}

__device__
Intersection intersection_create_no_intersect()
{
	Intersection inter;
	inter.is_intersect = 0;
	return inter;
}