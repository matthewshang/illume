#include "scene.h"

Scene* scene_new(int sphere_amount, int plane_amount)
{
	Scene* scene = (Scene *) calloc(sizeof(Scene), 1);
	if (!scene)
	{
		return NULL;
	}
	scene->sphere_amount = sphere_amount;
	scene->spheres = (Sphere *) calloc(sizeof(Sphere), sphere_amount);
	if (!scene->spheres)
	{
		free(scene);
		return NULL;
	}
	scene->plane_amount = plane_amount;
	scene->planes = (Plane *) calloc(sizeof(Plane), plane_amount);
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