#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kernel.h"
#include "bitmap.h"
#include "material.h"
#include "math/constants.h"
#include "math/vector3.h"
#include "math/transform.h"
#include "math/matrix4.h"
#include "primitives/sphere.h"
#include "primitives/plane.h"
#include "primitives/mesh.h"
#include "primitives/mesh_instance.h"
#include "scene/scenebuilder.h"
#include "scene/scene.h"

static char format[] = "%s-%sx%s-%sspp-%smd.png";

static Scene* init_scene()
{
	Material white = material_diffuse(vector3_create(0.75, 0.75, 0.75));
	Material blue = material_diffuse(vector3_create(0.25, 0.25, 0.75));
	Material red = material_diffuse(vector3_create(0.75, 0.25, 0.25));
	Material green = material_diffuse(vector3_create(0, 0.95, 0));
	Material mirror = material_specular(vector3_create(0.9, 0.9, 0.9));
	Material glass = material_refractive(vector3_create(0, 0, 0));
	SceneBuilder* builder = scenebuilder_new();

	scenebuilder_add_mesh(builder, mesh_new("res/cube.obj"));
	scenebuilder_add_mesh(builder, mesh_new("res/monkey.obj"));
	scenebuilder_add_mesh(builder, mesh_new("res/quad.obj"));
	scenebuilder_add_mesh(builder, mesh_new("res/bunny.obj"));

	//scenebuilder_add_mesh_instance(builder,
	//	mesh_instance_new(1, glass,
	//		transform_create(
	//			vector3_create(-0.5f, -0.15, 3), vector3_create(0.75, 0.75, 0.75),
	//				matrix4_mul(matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI),
	//						matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2)))));

	// Back
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, white,
			transform_create(
				vector3_create(0, 1.5, 7.5), vector3_create(10, 10, 10),
					matrix4_from_axis_angle(vector3_create(1, 0, 0), 0))));

	// Bottom
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, white,
			transform_create(
				vector3_create(0, -1, 5), vector3_create(10, 10, 10),
					matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	// Sides
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, blue,
			transform_create(
				vector3_create(3.0, 1.5, 5), vector3_create(10, 10, 10),
					matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI / 2))));

	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, red,
			transform_create(
				vector3_create(-3.0, 1.5, 5), vector3_create(10, 10, 10),
					matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI / 2))));
	
	// Ceiling
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, white,
			transform_create(
				vector3_create(0, 3.5, 5), vector3_create(10, 10, 10),
					matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(2, material_emissive(vector3_mul(vector3_create(3, 2.5, 1.5), 5)),
			transform_create(
				vector3_create(-0.0f, 3.5f - 0.0001f, 5.5f), vector3_create(2.25f, 1.25f, 1.0f),
					matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	//scenebuilder_add_sphere(builder, sphere_new(1, vector3_create(0, 4.5, 5), material_emissive(vector3_create(0.99 * 1.75, 0.99 * 1.75, 0.99 * 1.75))));
	
	//scenebuilder_add_mesh_instance(builder,
	//	mesh_instance_new(3, white,
	//		transform_create(
	//			vector3_create(0, 0, 5), vector3_create(1, 1, 1),
	//				matrix4_from_axis_angle(vector3_create(0, 1, 0), 0))));
	scenebuilder_add_sphere(builder, sphere_new(0.9, vector3_create(1.25, -0.1, 5), glass));
	scenebuilder_add_sphere(builder, sphere_new(0.9, vector3_create(-1.25, -0.1, 6), mirror));

	
	Scene* scene = scene_new(builder);
	scenebuilder_free(builder);
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
			//Camera camera = camera_create(vector3_create(0, 0, 0), 70, 4.5, 0.085);
			 Camera camera = camera_create(vector3_create(0, 1.0f, 0.5f), 90, 1, 0);

			render_scene(scene, image, camera, spp, max_depth);
			char* name = (char *)calloc(1 + _snprintf(NULL, 0, format, argv[1], argv[2], argv[3], argv[4], argv[5]), sizeof(char));
			sprintf(name, format, argv[1], argv[2], argv[3], argv[4], argv[5]);
			bitmap_save_to_png(image, name);
			printf("Saved to: %s\n", name);
			free(name);

			scene_free(scene);
		}
exit_scene:
		bitmap_free(image);
	}
exit_bitmap:
	return 0;
}