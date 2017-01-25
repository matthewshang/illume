#include "microfacet.h"

#include <math.h>

#include "math/mathutils.h"

__device__
float F_CookTorrance(Vector3 i, Vector3 m, float nt, float ni)
{
	float c = fabsf(vector3_dot(i, m));
	float nr = nt / ni;
	float g = nr * nr - 1.0f + c * c;
	if (g < 0) return 1;
	g = sqrtf(g);
	float a = (g - c) / (g + c);
	float b = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
	return 0.5f * a * a * (1 + b * b);
}

__device__ float F_dielectric(float cosI, float ior, float& cosT)
{
	float eta = cosI < 0.0f ? ior : 1.0f / ior;

	float sinTSq = eta * eta * (1.0f - cosI * cosI);
	if (sinTSq >= 1.0f)
	{
		cosT = 0.0f;
		return 1.0f;
	}
	cosI = fabsf(cosI);
	cosT = sqrtf(1.0f - sinTSq);
	float rperp = (eta * cosI - cosT) / (eta * cosI + cosT);
	float rpar = (cosI - eta * cosT) / (cosI + eta * cosT);
	return (rperp * rperp + rpar * rpar) * 0.5f;
}

__device__ float D_Beckmann(Vector3 m, Vector3 n, float alpha)
{
	float alphaSq = alpha * alpha;
	float cosT = vector3_dot(m, n);
	float cosTSq = cosT * cosT;
	float tanTSq = (cosTSq - 1.0f) / cosTSq;
	return ILLUME_INV_PI * expf(-tanTSq / alphaSq) / (alphaSq * cosTSq * cosTSq);
}

__device__ float pdf_Beckmann(Vector3 m, Vector3 n, float alpha)
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
float G_Beckmann(Vector3 i, Vector3 o, Vector3 n, float alpha)
{
	return G1_Beckmann(i, n, alpha) * G1_Beckmann(o, n, alpha);
}