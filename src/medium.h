#ifndef _MEDIUM_H_
#define _MEDIUM_H_

#include "math/vector3.h"

typedef struct Medium
{
	Vector3 absorption;
	float scattering;
	float g;
	bool active;
} Medium;

#ifdef __cplusplus
extern "C" {
#endif

Medium medium_create(Vector3 absorption, float scattering, float g);
__device__ __host__ Medium medium_air();

#ifdef __cplusplus
}
#endif

#endif