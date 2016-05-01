#include "mesh_instance.h"

MeshInstance* mesh_instance_new(int mesh_index, Vector3 pos)
{
	MeshInstance* instance = (MeshInstance *) calloc(1, sizeof(MeshInstance));
	if (!instance)
	{
		return NULL;
	}
	instance->mesh_index = mesh_index;
	instance->pos = pos;
	return instance;
}

void mesh_instance_free(MeshInstance* instance)
{
	if (instance)
	{
		free(instance);
	}
}

__device__  
Intersection mesh_instance_ray_intersect(MeshInstance instance, Mesh mesh, Ray ray)
{
	return mesh_ray_intersect(mesh, ray_create(vector3_sub(ray.o, instance.pos), ray.d));
}  
