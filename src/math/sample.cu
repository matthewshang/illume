#include "sample.h"

#include <math.h>

#include "mathutils.h"

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

// www.astro.umd.edu/~jph/HG_note.pdf
__device__
Vector3 sample_henyey_greenstein(float g, float u1, float u2)
{
	if (g == 0.0f)
	{
		return sample_sphere(u1, u2);
	}
	else
	{
		float phi = 2.0f * ILLUME_PI * u1;
		float frac = (1.0f - g * g) / (1.0f + g * (2.0f * u2 - 1));
		float cosT = (1.0f + g * g - frac * frac) / (2.0f * g);
		float sinT = sqrtf(fmaxf(1.0f - cosT, 0.0f));
		return vector3_create(cosf(phi) * sinT, cosT, sinf(phi) * sinT);
	}
}

__device__ Vector3 sample_beckmann(float a, float u1, float u2)
{
	float theta = atanf(sqrtf(-a * a * logf(1.0f - u1)));
	float phi = 2.0f * ILLUME_PI * u2;
	return vector3_create(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
}
