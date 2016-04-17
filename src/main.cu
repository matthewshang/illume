#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kernel.h"
#include "bitmap.h"
#include "material.h"
#include "math/vector3.h"
#include "primitives/sphere.h"
#include "primitives/plane.h"
#include "primitives/mesh.h"
#include "scene/scenebuilder.h"
#include "scene/scene.h"

static char format[] = "%s-%sx%s-%sspp-%smd.png";

static Scene* init_scene()
{
	Mesh mesh = mesh_new();
	mesh_load_obj(&mesh, "res/quad.obj");
	mesh_free(&mesh);
	SceneBuilder* builder = scenebuilder_new();
	Material white = material_diffuse(vector3_create(0.95, 0.95, 0.95));
	Material blue = material_diffuse(vector3_create(0, 0, 0.95));
	Material red = material_diffuse(vector3_create(0.95, 0, 0));
	Material green = material_diffuse(vector3_create(0, 0.95, 0));
	Material mirror = material_specular(vector3_create(0.9, 0.9, 0.9));

	scenebuilder_add_sphere(builder, sphere_new(0.75, vector3_create(0, -0.25, 7), mirror));
	scenebuilder_add_sphere(builder, sphere_new(0.35, vector3_create(1.5, -0.65, 5.5), red));
	scenebuilder_add_sphere(builder, sphere_new(0.25, vector3_create(-1, -0.75, 6), green));
	scenebuilder_add_sphere(builder, sphere_new(0.4, vector3_create(-1.5, -0.6, 5), blue));
	scenebuilder_add_sphere(builder, sphere_new(0.25, vector3_create(2, -0.75, 4), mirror));

	scenebuilder_add_plane(builder, plane_new(vector3_create(0, -1, 0), vector3_create(0, 1, 0), white));
	return scene_new(builder);
}

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		printf("illume: usage - <path> <width> <height> <spp> <maxdepth>\n");
		goto exit_bitmap;
	}
	{
		int width = strtol(argv[2], NULL, 10);
		int height = strtol(argv[3], NULL, 10);
		int spp = strtol(argv[4], NULL, 10);
		int max_depth = strtol(argv[5], NULL, 10);

		Bitmap* image = bitmap_new(width, height);
		if (!image)
		{
			goto exit_bitmap;
		}
		{
			Scene* scene = init_scene();
			if (!scene)
			{
				goto exit_scene;
			}
			Camera camera = camera_create(vector3_create(0, 0, 0), 70, 5.5, 0.2);
			render_scene(scene, image, camera, spp, max_depth);
			char name[snprintf(NULL, 0, format, argv[1], argv[2], argv[3], argv[4], argv[5])];
			sprintf(name, format, argv[1], argv[2], argv[3], argv[4], argv[5]);
			bitmap_save_to_png(image, name);
			printf("Saved to: %s\n", name);

			scene_free(scene);
		}
exit_scene:
		bitmap_free(image);
	}
exit_bitmap:
	return 0;
}