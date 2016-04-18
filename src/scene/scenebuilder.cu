#include "scenebuilder.h"

SceneBuilder* scenebuilder_new()
{
	SceneBuilder* builder = (SceneBuilder *) calloc(1, sizeof(SceneBuilder));
	if (!builder)
	{
		return NULL;
	}
	builder->spheres = arraylist_new(1);
	if (!builder->spheres)
	{
		goto exit;
	}
	builder->planes = arraylist_new(1);
	if (!builder->planes)
	{
		goto exit;
	}
	builder->meshes = arraylist_new(1);
	if (!builder->meshes)
	{
		goto exit;
	}
	return builder;
exit:
	scenebuilder_free(builder);
	return NULL;
}

void scenebuilder_free(SceneBuilder* builder)
{
	if (builder)
	{
		for (int i = 0; i < builder->spheres->length; i++)
		{
			sphere_free((Sphere *) arraylist_get(builder->spheres, i));
		}
		for (int i = 0; i < builder->planes->length; i++)
		{
			plane_free((Plane *) arraylist_get(builder->planes, i));
		}
		for (int i = 0; i < builder->meshes->length; i++)
		{
			Mesh* mesh = (Mesh *) arraylist_get(builder->meshes, i);
			if (mesh)
			{
				free(mesh);
			}
		}

		arraylist_free(builder->spheres);
		arraylist_free(builder->planes);
		arraylist_free(builder->meshes);
		free(builder);
	}
}

void scenebuilder_add_sphere(SceneBuilder* builder, Sphere* sphere)
{
	if (builder && sphere)
	{
		arraylist_add(builder->spheres, sphere);
	}
}

void scenebuilder_add_plane(SceneBuilder* builder, Plane* plane)
{
	if (builder && plane)
	{
		arraylist_add(builder->planes, plane);
	}
}

void scenebuilder_add_mesh(SceneBuilder* builder, Mesh* mesh)
{
	if (builder && mesh)
	{
		arraylist_add(builder->meshes, mesh);
	}
}