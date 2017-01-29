#pragma once

#include <math.h>

#include "math/vector3.h"
#include "intellisense.h"

namespace Fresnel
{
	__device__ 
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
}