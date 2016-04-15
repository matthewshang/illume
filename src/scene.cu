#include "scene.h"

Scene* scene_new(int sphere_amount, int plane_amount)
{
	Scene* scene = (Scene *) calloc(1, sizeof(Scene));
	if (!scene)
	{
		return NULL;
	}
	scene->sphere_amount = sphere_amount;
	scene->spheres = (Sphere *) calloc(sphere_amount, sizeof(Sphere));
	if (!scene->spheres)
	{
		free(scene);
		return NULL;
	}
	scene->plane_amount = plane_amount;
	scene->planes = (Plane *) calloc(plane_amount, sizeof(Plane));
	if (!scene->planes)
	{
		free(scene->spheres);
		free(scene);
		return NULL;
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

		free(scene);
	}
}