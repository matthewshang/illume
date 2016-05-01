#include "hit.h"

__device__
Hit hit_create(float d, Vector3 normal, Material m)
{
	Hit inter;
	inter.is_intersect = 1;
	inter.d = d;
	inter.normal = normal;
	inter.m = m;
	return inter;
}

__device__
Hit hit_create_no_intersect()
{
	Hit inter;
	inter.is_intersect = 0;
	return inter;
}