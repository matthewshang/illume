#include "ray.h"

__device__ __host__
Ray ray_create(Vector3 o, Vector3 d)
{
	Ray ray;
	ray.o = o;
	ray.d = d;
	vector3_normalize(&ray.d);
	return ray;
}

__device__ __host__
void ray_set(Ray* ray, Vector3 o, Vector3 d)
{
	if (ray)
	{
		vector3_set(&ray->o, o.x, o.y, o.z);
		vector3_set(&ray->d, d.x, d.y, d.z);
		vector3_normalize(&ray->d);
	}
}

__device__ __host__
Vector3 ray_position_along(Ray* ray, float d)
{
	if (ray)
	{
		Vector3 dmuld = vector3_mul(&ray->d, d);
		return vector3_add(&ray->o, &dmuld);
	}

	return vector3_create(0, 0, 0); 
}