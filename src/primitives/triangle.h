#ifndef _TRIANGLE_
#define _TRIANGLE_

#include "../math/vector3.h"

typedef struct 
{
	Vector3 n;
	Vector3 v0;
	Vector3 ex;
	Vector3 ey;
	float t1x;
	float t1y;
	float t2x;
	float t2y;
	float area;
}
Triangle;

#ifdef __cplusplus
extern "C" {
#endif

Triangle*  triangle_new     (Vector3 v0, Vector3 v1, Vector3 v2);
void       triangle_free    (Triangle* triangle);
Triangle   triangle_create  (Vector3 v0, Vector3 v1, Vector3 v2);

__device__ __host__
float      tri_area_times_two  (float ax, float ay, float bx, float by, float cx, float cy);

#ifdef __cplusplus
}
#endif

#endif