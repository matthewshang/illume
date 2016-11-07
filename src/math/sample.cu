#include "sample.h"

#include <math.h>

#include "constants.h"

__device__
Vector3 sample_hemisphere_cosine(float u1, float u2)
{
	float r = sqrtf(u1);
	float t = 2 * ILLUME_PI * u2;

	float x = r * cosf(t);
	float y = r * sinf(t);

	return vector3_create(x, sqrtf(1 - u1), y);
}

__device__
Vector3 sample_circle(float u1, float u2)
{
	float r = sqrtf(u1);
	float d = 2 * ILLUME_PI * u2;
	return vector3_create(r * cosf(d), r * sinf(d), 0);
}

__device__
Vector3 sample_sphere(float u1, float u2)
{
	float up = u1 * 2 - 1;
	float theta = 2 * ILLUME_PI * u2;
	float o = sqrtf(1.0f - up * up);
	return vector3_create(cosf(theta) * o, up, sinf(theta) * o);
}