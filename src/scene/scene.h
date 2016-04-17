#ifndef _SCENE_
#define _SCENE_

#include "../primitives/sphere.h"
#include "../primitives/plane.h"
#include "../primitives/mesh.h"
#include "scenebuilder.h"
#include "../arraylist.h"

typedef struct
{
	Sphere* spheres;
	int sphere_amount;
	Plane* planes;
	int plane_amount;
	Mesh* meshes;
	int mesh_amount;
} 
Scene;

#ifdef __cplusplus
extern "C" {
#endif

Scene*  scene_new   (SceneBuilder* builder);
void    scene_free  (Scene* scene);

#ifdef __cplusplus
}
#endif

#endif