#include "medium.h"

Medium medium_create(Vector3 absorption, float scattering, float g)
{
	Medium m;
	m.absorption = absorption;
	m.scattering = scattering;
	m.g = g;
	m.active = (vector3_length2(absorption) > 1e-5 || scattering > 0);
	return m;
}

__device__ __host__
Medium medium_air()
{
	Medium m;
	m.absorption = vector3_create(0, 0, 0);
	m.scattering = 0.0f;
	m.g = 0.0f;
	m.active = false;
	return m;
}