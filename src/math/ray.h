#ifndef _RAY_
#define _RAY_

#include "vector3.h"

typedef struct
{
	Vector3 o;
	Vector3 d;
}
Ray;

#ifdef __cplusplus
extern "C" {
#endif

__device__ __host__  Ray      ray_create          (Vector3 o, Vector3 d);
__device__ __host__  void     ray_set             (Ray* ray, Vector3 o, Vector3 d);
__device__ __host__  Vector3  ray_position_along  (Ray ray, float d);

#ifdef __cplusplus
}
#endif

#endif