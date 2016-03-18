#ifndef _RAY_
#define _RAY_

#include "vector3.h"

typedef struct
{
	Vector3 o;
	Vector3 d;
}
Ray;

__device__ __host__  Ray ray_create  (Vector3 o, Vector3 d);

#endif