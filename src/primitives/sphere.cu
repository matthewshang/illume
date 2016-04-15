#include "sphere.h"

Sphere sphere_create(float r, Vector3 center, Material m)
{
	Sphere sphere;
	sphere.r = r;
	sphere.center = center;
	sphere.m = m;
	return sphere;
}

__device__
Intersection sphere_ray_intersect(Sphere sphere, Ray ray)
{
	Vector3 l = vector3_sub(sphere.center, ray.o);
	float s = vector3_dot(l, ray.d);
	float ls = vector3_dot(l, l);
	float rs = sphere.r * sphere.r;
	if (s < 0 && ls > rs)
	{
		return intersection_create_no_intersect();
	}
	float ms = ls - s * s;
	if (ms > rs)
	{
		return intersection_create_no_intersect();
	}
	float q = sqrtf(rs - ms);
	float t = s;
	if (ls > rs)
	{
		t -= q;
	}
    else
	{
		t += q;
	}
	Vector3 pos = ray_position_along(ray, t);
	Vector3 normal = vector3_sub(pos, sphere.center);
	vector3_normalize(&normal);
	return intersection_create(t, normal, sphere.m);
}
