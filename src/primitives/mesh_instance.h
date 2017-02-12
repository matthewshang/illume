#ifndef _MESH_INSTANCE_
#define _MESH_INSTANCE_

#include "rapidjson/document.h"

#include "../math/ray.h"
#include "../math/transform.h"
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

			MeshInstance   mesh_instance_create         (int mesh_index, Material m, Transform t);
            MeshInstance*  mesh_instance_new            (int mesh_index, Material m, Transform t);
			MeshInstance   mesh_instance_from_json      (rapidjson::Value& json, Material m, int mesh_index, Mesh* mesh);
			void           mesh_instance_free           (MeshInstance* instance);
			void           mesh_instance_build_aabb     (MeshInstance* instance, Mesh* mesh);
__device__  void           mesh_instance_ray_intersect  (MeshInstance* instance, Mesh* mesh, Ray ray, Hit* hit);  

#endif