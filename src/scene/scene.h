#ifndef _SCENE_
#define _SCENE_

#include "../primitives/sphere.h"
#include "../primitives/mesh.h"
#include "../primitives/mesh_instance.h"
#include "scenebuilder.h"
#include "../camera.h"
#include "../math/vector3.h"

typedef struct
{
	Sphere* spheres;
	int sphere_amount;
	Mesh* meshes;
	int mesh_amount;
	MeshInstance* instances;
	int instance_amount;
	Camera camera;
	Vector3 sky_color;
} 
Scene;

#ifdef __cplusplus
extern "C" {
#endif

Scene*  scene_new   (const char* path);
void    scene_free  (Scene* scene);

#ifdef __cplusplus
}
#endif

#endif