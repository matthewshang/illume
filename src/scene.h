#ifndef _SCENE_
#define _SCENE_

#include "sphere.h"

typedef struct
{
	Sphere* spheres;
	int sphere_amount;
} 
Scene;

#ifdef __cplusplus
extern "C" {
#endif

Scene*  scene_new   (int sphere_amount);
void    scene_free  (Scene* scene);

#ifdef __cplusplus
}
#endif

#endif