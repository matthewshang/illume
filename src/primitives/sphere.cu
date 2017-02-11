#include "sphere.h"

#include "../jsonutils.h"
#include "../math/mathutils.h"

Sphere* sphere_new(float r, Vector3 center, Material m)
{
	Sphere* sphere = (Sphere *) calloc(1, sizeof(Sphere));
	if (!sphere)
	{
		return NULL;
	}
	*sphere = sphere_create(r, center, m);
	return sphere;
}

void sphere_free(Sphere* sphere)
{
	if (sphere)
	{
		free(sphere);
	}
}

Sphere sphere_from_json(rapidjson::Value& json, Material m)
{
	Sphere ret;
	JsonUtils::from_json(json, "radius", ret.r);
	JsonUtils::from_json(json, "center", ret.center);
	ret.m = m;
	return ret;
}

Sphere sphere_create(float r, Vector3 center, Material m)
{
	Sphere sphere;
	sphere.r = r;
	sphere.center = center;
	sphere.m = m;
	return sphere;
}

__device__
void sphere_ray_intersect(Sphere* sphere, Ray ray, Hit* hit)
{
	Vector3 l = vector3_sub(sphere->center, ray.o);
	float s = vector3_dot(l, ray.d);
	float ls = vector3_dot(l, l);
	float rs = sphere->r * sphere->r;
	if (s < 0 && ls > rs)
	{
		hit_set_no_intersect(hit);
		return;
	}
	float ms = ls - s * s;
	if (ms > rs)
	{
		hit_set_no_intersect(hit);
		return;
	}
	float q = sqrtf(rs - ms);
	float t = s;
	if (ls > rs)
	{
		t -= q;
	}
    else
	{
		t += q;
	}
	Vector3 pos = ray_position_along(ray, t);
	Vector3 normal = vector3_sub(pos, sphere->center);
	vector3_normalize(&normal);
    float u = atan2f(normal.z, normal.x) * 0.5f * ILLUME_INV_PI + 0.5f;
    float v = 0.5f - asinf(normal.y) * ILLUME_INV_PI;
    hit_set(hit, t, normal, &sphere->m, Vec2f(u, v));
}
