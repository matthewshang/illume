#ifndef _SCENE_
#define _SCENE_

#include "../primitives/sphere.h"
#include "../primitives/plane.h"
#include "../primitives/mesh.h"
#include "../primitives/mesh_instance.h"
#include "scenebuilder.h"
#include "../camera.h"

typedef struct
{
	Sphere* spheres;
	int sphere_amount;
	Plane* planes;
	int plane_amount;
	Mesh* meshes;
	int mesh_amount;
	MeshInstance* instances;
	int instance_amount;
	Camera camera;
} 
Scene;

#ifdef __cplusplus
extern "C" {
#endif

Scene*  scene_new   (SceneBuilder* builder, Camera camera);
void    scene_free  (Scene* scene);

#ifdef __cplusplus
}
#endif

#endif