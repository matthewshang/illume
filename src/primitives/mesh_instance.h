#ifndef _MESH_INSTANCE_
#define _MESH_INSTANCE_

#include "../math/vector3.h"
#include "../math/ray.h"
#include "../math/transform.h"
#include "../math/matrix4.h"
#include "../math/aabb.h"
#include "mesh.h"
#include "../material.h"
#include "../hit.h"

typedef struct
{
	int mesh_index;
	Material m;
	Transform t;
	AABB aabb;
}
MeshInstance;

#ifdef __cplusplus
extern "C" {
#endif

            MeshInstance*  mesh_instance_new            (int mesh_index, Material m, Transform t);
			void           mesh_instance_free           (MeshInstance* instance);
			void           mesh_instance_build_aabb     (MeshInstance* instance, Mesh mesh);
__device__  Hit            mesh_instance_ray_intersect  (MeshInstance* instance, Mesh mesh, Ray ray);  

#ifdef __cplusplus
}
#endif

#endif