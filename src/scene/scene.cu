#include "scene.h"

#include "../arraylist.h"

Scene* scene_new(SceneBuilder* builder)
{
	Scene* scene = (Scene *) calloc(1, sizeof(Scene));
	if (!scene)
	{
		return NULL;
	}
	int sphere_amount = builder->spheres->length;
	int plane_amount = builder->planes->length;
	int mesh_amount = builder->meshes->length;
	int instance_amount = builder->instances->length;

	scene->sphere_amount = sphere_amount;
	scene->spheres = (Sphere *) calloc(sphere_amount, sizeof(Sphere));
	if (!scene->spheres)
	{
		scene_free(scene);
		return NULL;
	}
	scene->plane_amount = plane_amount;
	scene->planes = (Plane *) calloc(plane_amount, sizeof(Plane));
	if (!scene->planes)
	{
		scene_free(scene);
		return NULL;
	}
	scene->mesh_amount = mesh_amount;
	scene->meshes = (Mesh *) calloc(mesh_amount, sizeof(Mesh));
	if (!scene->meshes)
	{
		scene_free(scene);
		return NULL;
	}
	scene->instance_amount = instance_amount;
	scene->instances = (MeshInstance *) calloc(instance_amount, sizeof(MeshInstance));
	if (!scene->instances)
	{
		scene_free(scene);
		return NULL;
	}
	for (int i = 0; i < sphere_amount; i++)
	{
		scene->spheres[i] = *((Sphere *) arraylist_get(builder->spheres, i));
	}
	for (int i = 0; i < plane_amount; i++)
	{
		scene->planes[i] = *((Plane *) arraylist_get(builder->planes, i));
	}
	for (int i = 0; i < mesh_amount; i++)
	{
		scene->meshes[i] = *((Mesh *) arraylist_get(builder->meshes, i));
	}
	for (int i = 0; i < instance_amount; i++)
	{
		scene->instances[i] = *((MeshInstance *) arraylist_get(builder->instances, i));
		mesh_instance_build_aabb(&scene->instances[i], scene->meshes[scene->instances[i].mesh_index]);	
	}
	return scene;
}

void scene_free(Scene* scene)
{
	if (scene)
	{
		if (scene->spheres)
		{
			free(scene->spheres);
		}

		if (scene->planes)
		{
			free(scene->planes);
		}

		if (scene->instances)
		{
			free(scene->instances);
		}

		if (scene->meshes)
		{
			for (int i = 0; i < scene->mesh_amount; i++)
			{
				if (scene->meshes[i].triangles)
				{
					free(scene->meshes[i].triangles);
				}
				bvh_free(&scene->meshes[i].bvh);
			}
			free(scene->meshes);
		}

		free(scene);
	}
}