#ifndef _MICROFACET_
#define _MICROFACET_

#include "math/vector3.h"

#ifdef __cplusplus
	extern "C" {
#endif

	__device__ float F_CookTorrance(Vector3 i, Vector3 m, float nt, float ni);
	__device__ float F_dielectric(float cosI, float ior, float& cosT);

	__device__ float D_Beckmann(Vector3 m, Vector3 n, float alpha);
	__device__ float pdf_Beckmann(Vector3 m, Vector3 n, float alpha);

	__device__ float G_Beckmann(Vector3 i, Vector3 o, Vector3 n, float alpha);
#ifdef __cplusplus
	}
#endif

#endif