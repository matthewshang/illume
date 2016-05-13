#include "aabb.h"

AABB aabb_create()
{
	AABB aabb;
	aabb.min = vector3_create(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);
	aabb.max = vector3_create(FLOAT_MIN, FLOAT_MIN, FLOAT_MIN);
	return aabb;
}

AABB aabb_from_vertices(Vector3 min, Vector3 max)
{
	AABB aabb;
	aabb.min = min;
	aabb.max = max;
	return aabb;
}

void aabb_update(AABB* aabb, Vector3 p)
{
	if (aabb)
	{
		aabb->min.x = fminf(aabb->min.x, p.x);
		aabb->min.y = fminf(aabb->min.y, p.y);
		aabb->min.z = fminf(aabb->min.z, p.z);
		aabb->max.x = fmaxf(aabb->max.x, p.x);
		aabb->max.y = fmaxf(aabb->max.y, p.y);
		aabb->max.z = fmaxf(aabb->max.z, p.z);
	}
}

void aabb_get_vertices(AABB aabb, Vector3* vertices)
{
	Vector3 min = aabb.min;
	Vector3 max = aabb.max;
	vertices[0] = min;
	vertices[1] = vector3_create(max.x, min.y, min.z);
	vertices[2] = vector3_create(max.x, max.y, min.z);
	vertices[3] = vector3_create(min.x, max.y, min.z);
	vertices[4] = vector3_create(min.x, min.y, max.z);
	vertices[5] = vector3_create(max.x, min.y, max.z);
	vertices[6] = max;
	vertices[7] = vector3_create(min.x, max.y, max.z);
}

__device__
float aabb_ray_intersect(AABB aabb, Ray ray)
{
	Vector3 o = ray.o;
	Vector3 d = ray.d;
	Vector3 min = aabb.min;
	Vector3 max = aabb.max;
	float tmin = FLOAT_MIN;
	float tmax = FLOAT_MAX;

	if (d.x != 0)
	{
		float t1 = (min.x - o.x) / d.x;
		float t2 = (max.x - o.x) / d.x;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (d.y != 0)
	{
		float t1 = (min.y - o.y) / d.y;
		float t2 = (max.y - o.y) / d.y;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (d.z != 0)
	{
		float t1 = (min.z - o.z) / d.z;
		float t2 = (max.z - o.z) / d.z;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}

	if (tmax >= tmin && tmax >= 0)
	{
		return tmin;
	}
	return -1;
}

int aabb_aabb_intersect(AABB u, AABB v)
{
	return !(u.min.x > v.max.x || u.max.x < v.min.x ||
			 u.min.y > v.max.y || u.max.y < v.min.y ||
			 u.min.z > v.max.z || u.max.z < v.min.z);
}