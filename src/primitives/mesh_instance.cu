#include "mesh_instance.h"

MeshInstance* mesh_instance_new(int mesh_index, Material m, Transform t)
{
	MeshInstance* instance = (MeshInstance *) calloc(1, sizeof(MeshInstance));
	if (!instance)
	{
		return NULL;
	}
	instance->mesh_index = mesh_index;
	instance->m = m;
	instance->t = t;
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
Hit mesh_instance_ray_intersect(MeshInstance* instance, Mesh mesh, Ray ray)
{
	Vector3 new_origin = matrix4_mul_vector3(&instance->t.inv, ray.o, 1);
	Vector3 new_dir = matrix4_mul_vector3(&instance->t.trans, ray.d, 0);
	Ray new_ray = ray_create(new_origin, new_dir);
	Hit isect = mesh_ray_intersect(mesh, new_ray);

	if (isect.is_intersect)
	{
		Vector3 p = matrix4_mul_vector3(&instance->t.mat, ray_position_along(new_ray, isect.d), 1);
		isect.d = vector3_length(vector3_sub(ray.o, p));
		isect.normal = matrix4_mul_vector3(&instance->t.trans_inv, isect.normal, 0);
		vector3_normalize(&isect.normal);
		isect.m = instance->m;
	}

	return isect;
}  
