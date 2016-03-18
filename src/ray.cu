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