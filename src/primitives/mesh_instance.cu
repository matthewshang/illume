#include "mesh_instance.h"

#include <stdlib.h>
#include <stdio.h>

#include "../math/vector3.h"
#include "../math/matrix4.h"
#include "../math/constants.h"

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
		AABB* inst_aabb = &instance->aabb;

		for (int i = 0; i < 8; i++)
		{
			aabb_update(inst_aabb, matrix4_mul_vector3(&instance->t.mat, aabb_vertices[i], 1));
		}
		float bias = 10 * FLT_EPSILON;
		if (inst_aabb->max.x - inst_aabb->min.x < bias)
		{
			inst_aabb->max.x += bias;
		}
		if (inst_aabb->max.y - inst_aabb->min.y < bias)
		{
			inst_aabb->max.y += bias;
		}
		if (inst_aabb->max.z - inst_aabb->min.z < bias)
		{
			inst_aabb->max.z += bias;
		}
		printf("mesh instance bounds: \n  min: %f %f %f\n  max: %f %f %f\n", inst_aabb->min.x, inst_aabb->min.y, inst_aabb->min.z, inst_aabb->max.x, inst_aabb->max.y, inst_aabb->max.z);
	}
}

__device__  
void mesh_instance_ray_intersect(MeshInstance* instance, Mesh* mesh, Ray ray, Hit* hit)
{
	if (aabb_ray_intersect(instance->aabb, ray) == -FLT_MAX)
	{
		hit_set_no_intersect(hit);
	}

	Vector3 new_origin = matrix4_mul_vector3(&instance->t.inv, ray.o, 1);
	Vector3 new_dir = matrix4_mul_vector3(&instance->t.inv, ray.d, 0);
	Ray new_ray;
	new_ray.o = new_origin;
	new_ray.d = new_dir;
	mesh_ray_intersect(mesh, new_ray, hit);
	if (hit->is_intersect)
	{
		hit->normal = matrix4_mul_vector3(&instance->t.trans_inv, hit->normal, 0);
		vector3_normalize(&hit->normal);
		hit->m = instance->m;
	}
}  
