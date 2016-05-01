#ifndef _MESH_INSTANCE_
#define _MESH_INSTANCE_

#include "../math/vector3.h"
#include "mesh.h"
#include "../math/ray.h"

typedef struct
{
	int mesh_index;
	Vector3 pos;
}
MeshInstance;

#ifdef __cplusplus
extern "C" {
#endif

            MeshInstance*  mesh_instance_new            (int mesh_index, Vector3 pos);
			void           mesh_instance_free           (MeshInstance* instance);
__device__  Intersection   mesh_instance_ray_intersect  (MeshInstance instance, Mesh mesh, Ray ray);  

#ifdef __cplusplus
}
#endif

#endif