#include "intersection.h"

__device__
Intersection intersection_create(int is_intersect)
{
	Intersection inter;
	inter.is_intersect = is_intersect;
	return inter;
}