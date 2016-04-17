#ifndef _SCENEBUILDER_
#define _SCENEBUILDER_

#include "../arraylist.h"
#include "../primitives/mesh.h"
#include "../primitives/plane.h"
#include "../primitives/sphere.h"

typedef struct
{
	ArrayList* spheres;
	ArrayList* planes;
	ArrayList* meshes;
}
SceneBuilder;

#ifdef __cplusplus
extern "C" {
#endif

SceneBuilder*  scenebuilder_new         ();
void           scenebuilder_free        (SceneBuilder* builder);
void           scenebuilder_add_sphere  (SceneBuilder* builder, Sphere* sphere);
void           scenebuilder_add_plane   (SceneBuilder* builder, Plane* plane);
void           scenebuilder_add_mesh    (SceneBuilder* builder, Mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif