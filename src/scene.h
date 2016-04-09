#ifndef _SCENE_
#define _SCENE_

#include "sphere.h"
#include "plane.h"

typedef struct
{
	Sphere* spheres;
	int sphere_amount;
	Plane* planes;
	int plane_amount;
} 
Scene;

#ifdef __cplusplus
extern "C" {
#endif

Scene*  scene_new   (int sphere_amount, int plane_amount);
void    scene_free  (Scene* scene);

#ifdef __cplusplus
}
#endif

#endif