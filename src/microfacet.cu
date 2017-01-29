#include "microfacet.h"

#include <math.h>

#include "math/mathutils.h"

__device__ 
float Microfacet::D_Beckmann(Vector3 m, Vector3 n, float alpha)
{
	float alphaSq = alpha * alpha;
	float cosT = vector3_dot(m, n);
	float cosTSq = cosT * cosT;
	float tanTSq = (cosTSq - 1.0f) / cosTSq;
	return ILLUME_INV_PI * expf(-tanTSq / alphaSq) / (alphaSq * cosTSq * cosTSq);
}

__device__ 
float Microfacet::pdf_Beckmann(Vector3 m, Vector3 n, float alpha)
{
	return D_Beckmann(m, n, alpha) * fabsf(vector3_dot(m, n));
}

__device__
float G1_Beckmann(Vector3 v,  Vector3 n, float alpha)
{
	float cosV = vector3_dot(v, n);
	float tanV = sqrtf(1.0f - cosV * cosV) / cosV;
	float a = 1.0f / (alpha * fabsf(tanV));
	if (a < 1.6)
	{
		return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
	}
	else
	{
		return 1.0f;
	}
}

__device__
float Microfacet::G_Beckmann(Vector3 i, Vector3 o, Vector3 n, float alpha)
{
	return G1_Beckmann(i, n, alpha) * G1_Beckmann(o, n, alpha);
}


__device__ 
Vector3 Microfacet::sample_Beckmann(float a, float u1, float u2)
{
	float theta = atanf(sqrtf(-a * a * logf(1.0f - u1)));
	float phi = 2.0f * ILLUME_PI * u2;
	return vector3_create(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
}