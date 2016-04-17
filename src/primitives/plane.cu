#include "plane.h"

Plane* plane_new(Vector3 p, Vector3 n, Material m)
{
	Plane* plane = (Plane *) calloc(1, sizeof(Plane));
	if (!plane)
	{
		return NULL;
	}
	*plane = plane_create(p, n, m);
	return plane;
}

void plane_free(Plane* plane)
{
	if (plane)
	{
		free(plane);
	}
}

Plane plane_create(Vector3 p, Vector3 n, Material m)
{
	Plane plane;
	plane.p = p;
	plane.n = n;
	plane.m = m;
	return plane;
}

__device__
Intersection plane_ray_intersect(Plane plane, Ray r)
{
	float d = vector3_dot(plane.n, vector3_sub(plane.p, r.o)) / vector3_dot(plane.n, r.d);
	if (d < 0)
	{
		return intersection_create_no_intersect();
	}

	return intersection_create(d, plane.n, plane.m);
}