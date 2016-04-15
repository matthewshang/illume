#include "sample.h"

__device__
Vector3 sample_hemisphere_cosine(float u1, float u2)
{
	float r = sqrtf(u1);
	float t = 2 * PI * u2;

	float x = r * cosf(t);
	float y = r * sinf(t);

	return vector3_create(x, sqrtf(1 - u1), y);
}

__device__
Vector3 sample_circle(float u1, float u2)
{
	float r = sqrtf(u1);
	float d = 2 * PI * u2;
	return vector3_create(r * cosf(d), r * sinf(d), 0);
}