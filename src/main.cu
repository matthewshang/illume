#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kernel.h"
#include "bitmap.h"
#include "scene.h"
#include "vector3.h"
#include "material.h"
#include "sphere.h"

static char format[] = "%s-%sx%s-%sspp-%smd.png";

static Scene* init_scene()
{
	Material white = material_diffuse(vector3_create(0.95, 0.95, 0.95));
	Material blue = material_diffuse(vector3_create(0, 0, 0.85));
	Material red = material_diffuse(vector3_create(0.85, 0, 0));

	Scene* scene = scene_new(4);
	scene->spheres[0] = sphere_create(10, vector3_create(0, -11, 8), white);
	scene->spheres[1] = sphere_create(1, vector3_create(0, 0, 8), white);
	scene->spheres[2] = sphere_create(0.5, vector3_create(-2, -0.75, 7), red);
	scene->spheres[3] = sphere_create(0.5, vector3_create(2, -0.75, 7), blue);
	return scene;
}

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		printf("illume: usage - <path> <width> <height> <spp> <maxdepth>\n");
		goto exit_bitmap;
	}
	{
		int width = atoi(argv[2]);
		int height = atoi(argv[3]);
		int spp = atoi(argv[4]);
		int max_depth = atoi(argv[5]);

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
			render_scene(scene, image, spp, max_depth);
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