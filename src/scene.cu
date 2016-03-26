#include "scene.h"

Scene* scene_new(int sphere_amount)
{
	Scene* scene = (Scene *) calloc(sizeof(Scene), 1);
	if (!scene)
	{
		return NULL;
	}
	scene->sphere_amount = sphere_amount;
	scene->spheres = (Sphere *) calloc(sizeof(Sphere) * scene->sphere_amount, 1);
	if (!scene->spheres)
	{
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

		free(scene);
	}
}