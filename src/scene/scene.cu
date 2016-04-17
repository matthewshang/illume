#include "scene.h"

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

		if (scene->meshes)
		{
			free(scene->meshes);
		}

		free(scene);
	}
}