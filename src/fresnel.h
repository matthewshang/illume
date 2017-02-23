#pragma once

#include <math.h>

#include "math/vector3.h"
#include "intellisense.h"

namespace Fresnel
{
	__device__ __host__
	inline float dielectric(float cosI, float ior, float& cosT)
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

	// seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/#more-1921
	__device__
	inline float conductor(float eta, float k, float cosI)
	{
		float cosISq = cosI * cosI;
		float sinISq = 1 - cosISq;
		float etaSq = eta * eta;
		float kSq = k * k;

		float t0 = etaSq - kSq - sinISq;
		float a2pb2 = sqrtf(t0 * t0 + 4.0f * etaSq * kSq);
		float t1 = a2pb2 + cosISq;
		float a = sqrtf(0.5f * (a2pb2 + t0));
		float t2 = 2.0f * a * cosI;
		float Rs = (t1 - t2) / (t1 + t2);

		float t3 = cosISq * a2pb2 + sinISq * sinISq;
		float t4 = t2 * sinISq;
		float Rp = Rs * (t3 - t4) / (t3 + t4);

		return 0.5f * (Rp + Rs);
	}

	__device__
	inline Vector3 conductor(Vector3 eta, Vector3 k, float cosI)
	{
		return vector3_create(conductor(eta.x, k.x, cosI),
							  conductor(eta.y, k.y, cosI),
							  conductor(eta.z, k.z, cosI));
	}

    // Approximates the integral over the hemisphere for Fresnel::dielectric, from Tungsten
    inline float diffuse_fresnel(float ior, int samples)
    {
        float cosT;
        double ret = 0.0;
        float fb = dielectric(0.0, ior, cosT);
        for (int i = 1; i <= samples; i++)
        {
            float cosTSq = (float) i / samples;
            float fa = dielectric(ior, sqrtf(cosTSq), cosT);
            ret += (double)(fa + fb) * (0.5 / samples);
            fb = fa;
        }
        return (float)ret;
    }
}