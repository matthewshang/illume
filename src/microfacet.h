#pragma once

#include "math/vector3.h"

namespace Microfacet
{
	__device__ float D_Beckmann(Vector3 m, Vector3 n, float alpha);
	__device__ float pdf_Beckmann(Vector3 m, Vector3 n, float alpha);

	__device__ float G_Beckmann(Vector3 i, Vector3 o, Vector3 n, float alpha);

	__device__ Vector3 sample_Beckmann(float a, float u1, float u2);
}