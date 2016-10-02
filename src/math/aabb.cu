#include "aabb.h"

#include "constants.h"

AABB aabb_create()
{
	AABB aabb;
	aabb.min = vector3_create(FLT_MAX, FLT_MAX, FLT_MAX);
	aabb.max = vector3_create(-FLT_MAX, -FLT_MAX, -FLT_MAX);
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

__device__ __host__
void aabb_ray_get_points(AABB aabb, Ray ray, float* entry, float* exit)
{
	float tmin = -FLT_MAX;
	float tmax = FLT_MAX;

	if (ray.d.x != 0)
	{
		float t1 = (aabb.min.x - ray.o.x) / ray.d.x;
		float t2 = (aabb.max.x - ray.o.x) / ray.d.x;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.y != 0)
	{
		float t1 = (aabb.min.y - ray.o.y) / ray.d.y;
		float t2 = (aabb.max.y - ray.o.y) / ray.d.y;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.z != 0)
	{
		float t1 = (aabb.min.z - ray.o.z) / ray.d.z;
		float t2 = (aabb.max.z - ray.o.z) / ray.d.z;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}

	if (tmax >= tmin && tmax > 0)
	{
		*entry = tmin;
		*exit = tmax;
	}
}

__device__
float aabb_ray_exit(AABB aabb, Ray ray)
{
	float tmin = -FLT_MAX;
	float tmax = FLT_MAX;

	if (ray.d.x != 0)
	{
		float t1 = (aabb.min.x - ray.o.x) / ray.d.x;
		float t2 = (aabb.max.x - ray.o.x) / ray.d.x;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.y != 0)
	{
		float t1 = (aabb.min.y - ray.o.y) / ray.d.y;
		float t2 = (aabb.max.y - ray.o.y) / ray.d.y;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.z != 0)
	{
		float t1 = (aabb.min.z - ray.o.z) / ray.d.z;
		float t2 = (aabb.max.z - ray.o.z) / ray.d.z;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}

	if (tmax >= tmin && tmax >= 0)
	{
		return tmax;
	}
	return -FLT_MAX;
}

__device__
float aabb_ray_intersect(AABB aabb, Ray ray)
{
	float tmin = -FLT_MAX;
	float tmax = FLT_MAX;

	if (ray.d.x != 0)
	{
		float t1 = (aabb.min.x - ray.o.x) / ray.d.x;
		float t2 = (aabb.max.x - ray.o.x) / ray.d.x;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.y != 0)
	{
		float t1 = (aabb.min.y - ray.o.y) / ray.d.y;
		float t2 = (aabb.max.y - ray.o.y) / ray.d.y;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}
	if (ray.d.z != 0)
	{
		float t1 = (aabb.min.z - ray.o.z) / ray.d.z;
		float t2 = (aabb.max.z - ray.o.z) / ray.d.z;
		tmin = fmaxf(tmin, fminf(t1, t2));
		tmax = fminf(tmax, fmaxf(t1, t2));
	}

	if (tmax > tmin && tmax >= 0)
	{
		return tmin;
	}
	return -FLT_MAX;
}

int aabb_aabb_intersect(AABB u, AABB v)
{
	return !(u.min.x > v.max.x || u.max.x < v.min.x ||
			 u.min.y > v.max.y || u.max.y < v.min.y ||
			 u.min.z > v.max.z || u.max.z < v.min.z);
}