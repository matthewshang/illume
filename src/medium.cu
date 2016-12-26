#include "medium.h"

#include "jsonutils.h"

Medium medium_from_json(rapidjson::Value& json)
{
	Medium ret;
	JsonUtils::from_json(json, "absorption", ret.absorption);
	ret.absorption = vector3_sub(vector3_create(1, 1, 1), ret.absorption);
	JsonUtils::from_json(json, "scattering", ret.scattering);
	JsonUtils::from_json(json, "g",          ret.g);
	ret.active = (vector3_length2(ret.absorption) > 1e-5 || ret.scattering > 0);
	return ret;
}

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