#ifndef _SPHERE_
#define _SPHERE_

#include "rapidjson/document.h"

#include "../math/vector3.h"
#include "../math/ray.h"
#include "../hit.h"
#include "../material.h"

typedef struct
{
	float r;
	Vector3 center;
	Material m;
} 
Sphere;

#ifdef __cplusplus
extern "C" {
#endif
            Sphere*  sphere_new            (float r, Vector3 center, Material m);
            void     sphere_free           (Sphere* sphere);
			Sphere   sphere_from_json      (rapidjson::Value& json, Material m);
			Sphere   sphere_create         (float r, Vector3 center, Material m);
__device__  void     sphere_ray_intersect  (Sphere* sphere, Ray ray, Hit* hit);

#ifdef __cplusplus
}
#endif

#endif