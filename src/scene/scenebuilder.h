#ifndef _SCENEBUILDER_
#define _SCENEBUILDER_

#include "../arraylist.h"
#include "../primitives/mesh.h"
#include "../primitives/sphere.h"
#include "../primitives/mesh_instance.h"

typedef struct
{
	ArrayList* spheres;
	ArrayList* planes;
	ArrayList* meshes;
	ArrayList* instances;
}
SceneBuilder;

#ifdef __cplusplus
extern "C" {
#endif

SceneBuilder*  scenebuilder_new                ();
void           scenebuilder_free               (SceneBuilder* builder);
void           scenebuilder_add_sphere         (SceneBuilder* builder, Sphere* sphere);
void           scenebuilder_add_mesh           (SceneBuilder* builder, Mesh* mesh);
void           scenebuilder_add_mesh_instance  (SceneBuilder* builder, MeshInstance* mesh);

#ifdef __cplusplus
}
#endif

#endif