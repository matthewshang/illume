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
	instance->aabb = aabb_create();
	return instance;
}

void mesh_instance_free(MeshInstance* instance)
{
	if (instance)
	{
		free(instance);
	}
}

void mesh_instance_build_aabb(MeshInstance* instance, Mesh mesh)
{
	if (instance)
	{
		Vector3 aabb_vertices[8];
		aabb_get_vertices(mesh.aabb, aabb_vertices);
		for (int i = 0; i < 8; i++)
		{
			aabb_update(&instance->aabb, matrix4_mul_vector3(&instance->t.mat, aabb_vertices[i], 1));
		}
	}
}

__device__  
Hit mesh_instance_ray_intersect(MeshInstance* instance, Mesh* mesh, Ray ray)
{
	if (aabb_ray_intersect(instance->aabb, ray) == -1)
	{
		return hit_create_no_intersect();
	}

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
