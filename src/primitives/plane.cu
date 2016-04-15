#include "plane.h"

Plane plane_create(Vector3 p, Vector3 n, Material m)
{
	Plane plane;
	plane.p = p;
	plane.n = n;
	plane.m = m;
	return plane;
}

__device__
Intersection plane_ray_intersect(Plane p, Ray r)
{
	float d = vector3_dot(p.n, vector3_sub(p.p, r.o)) / vector3_dot(p.n, r.d);
	if (d < 0)
	{
		return intersection_create_no_intersect();
	}

	return intersection_create(d, p.n, p.m);
}