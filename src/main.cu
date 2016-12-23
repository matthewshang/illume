#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kernel.h"
#include "bitmap.h"
#include "material.h"
#include "medium.h"
#include "math/constants.h"
#include "math/vector3.h"
#include "math/transform.h"
#include "math/matrix4.h"
#include "primitives/sphere.h"
#include "primitives/mesh.h"
#include "primitives/mesh_instance.h"
#include "scene/scenebuilder.h"
#include "scene/scene.h"

static char format[] = "%s-%sx%s-%sspp-%smd.png";

static Scene* init_scene()
{
	Medium sss_orange = medium_create(vector3_create(1.0f - 0.9999f, 1.0f - 0.2758f, 1.0f - 0.0278f), 8.0f, 0.0f);
	Medium sss_jade = medium_create(vector3_create(1.0f - 0.2758f, 1.0f - 0.9599f, 1.0f - 0.0278f), 24.0f, -0.4f);

	Material white = material_diffuse(vector3_create(0.75, 0.75, 0.75));
	Material marble = material_diffuse(vector3_create(0.7968, 0.7815, 0.6941));
	Material ground = material_diffuse(vector3_create(0.4, 0.4, 0.4));
	Material blue = material_diffuse(vector3_create(0.25, 0.25, 0.75));
	Material red = material_diffuse(vector3_create(0.75, 0.25, 0.25));
	Material mirror = material_specular(vector3_create(0.99, 0.99, 0.99));
	Material glass = material_refractive(vector3_create(0.99f, 0.99f, 0.99f), 1.5f, medium_air());
	//Material glass = material_refractive(vector3_create(0.7893f, 0.6121f, 0.9999f), 1.68f, sss_jade);
	Material glossy = material_cooktorrance(vector3_create(0.99, 0.99, 0.99), 1.5f, 0.4f);

	SceneBuilder* builder = scenebuilder_new();

	scenebuilder_add_mesh(builder, mesh_new("res/quad.obj", 0, 4));
	// Back
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, white,
			transform_create(
				vector3_create(0, 1.5, 7.5), vector3_create(10, 10, 10),
				matrix4_from_axis_angle(vector3_create(1, 0, 0), 0))));

	// Bottom
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, white,
			transform_create(
				vector3_create(0, -1, 5), vector3_create(10, 10, 10),
				matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	// Sides
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, blue,
			transform_create(
				vector3_create(3.0, 1.5, 5), vector3_create(10, 10, 10),
				matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI / 2))));

	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, red,
			transform_create(
				vector3_create(-3.0, 1.5, 5), vector3_create(10, 10, 10),
				matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI / 2))));

	// Ceiling
	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, white,
			transform_create(
				vector3_create(0, 3.5, 5), vector3_create(10, 10, 10),
				matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(0, material_emissive(vector3_mul(vector3_create(3, 2.5, 1.5), 5)),
			transform_create(
				vector3_create(-0.0f, 3.5f - 0.0001f, 5.f), vector3_create(2.25f, 1.25f, 1.0f),
				matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));

	//scenebuilder_add_mesh_instance(builder,
	//	mesh_instance_new(0, material_cooktorrance(vector3_create(0.99f, 0.99f, 0.99f), 1.5f, 0.15f),
	//		transform_create(vector3_create(0.0f, 1.25f, 7.499), vector3_create(5.0f, 3.5f, 1.0f),
	//			matrix4_mul(matrix4_from_axis_angle(vector3_create(1, 0, 0), 0),
	//						matrix4_from_axis_angle(vector3_create(0, 1, 0), 0)))));

	scenebuilder_add_mesh(builder, mesh_new("res/xyzrgb_dragon.obj", 1, 8));

	scenebuilder_add_mesh_instance(builder,
		mesh_instance_new(1, material_cooktorrance(vector3_create(0.0290f, 0.6121f, 0.6121f), 1.59f, 0.2f),
			transform_create(vector3_create(0.295f, -1.01f, 5.0f), vector3_create(0.25f, 0.25f, 0.25f),
				matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI * -130.f / 180.f))));

	//scenebuilder_add_mesh_instance(builder,
	//	mesh_instance_new(1, material_cooktorrance(vector3_create(0.0290f, 0.6121f, 0.6121f), 1.59f, 0.15f),
	//		transform_create(vector3_create(0.15f, -1.01f, 5.0f), vector3_create(0.25f, 0.25f, 0.25f),
	//			matrix4_from_axis_angle(vector3_create(0, 1, 0), ILLUME_PI * -130.f / 180.f))));


	Scene* scene = scene_new("res/testscene/scene.json");

	scenebuilder_free(builder);
	return scene;
}

//static Scene* init_scene()
//{
//	Medium sss_orange = medium_create(vector3_create(1.0f - 0.9999f, 1.0f - 0.2758f, 1.0f - 0.0278f), 8.0f, 0.0f);
//	Medium sss_jade = medium_create(vector3_create(1.0f - 0.2758f, 1.0f - 0.9599f, 1.0f - 0.0278f), 24.0f, -0.4f);
//
//	Material white = material_diffuse(vector3_create(0.75, 0.75, 0.75));
//	Material marble = material_diffuse(vector3_create(0.7968, 0.7815, 0.6941));
//	Material ground = material_diffuse(vector3_create(0.4, 0.4, 0.4));
//	Material blue = material_diffuse(vector3_create(0.25, 0.25, 0.75));
//	Material red = material_diffuse(vector3_create(0.75, 0.25, 0.25));
//	Material mirror = material_specular(vector3_create(0.99, 0.99, 0.99));
//	//Material glass = material_refractive(vector3_create(0.99f, 0.99f, 0.99f), 1.5f);
//	Material glass = material_refractive(vector3_create(0.7893f, 0.6121f, 0.9999f), 1.68f, sss_jade);
//
//	SceneBuilder* builder = scenebuilder_new();
//
//
//	scenebuilder_add_mesh(builder, mesh_new("res/dragon.obj", 1, 4));
//	scenebuilder_add_mesh_instance(builder,
//		mesh_instance_new(0, glass,
//			transform_create(
//				vector3_create(0, -2, 7), vector3_create(0.75, 0.75, 0.75),
//				matrix4_mul(matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / -2),
//					matrix4_from_axis_angle(vector3_create(0, 0, 1), ILLUME_PI / -18)))));
//
//	scenebuilder_add_mesh(builder, mesh_new("res/quad.obj", 0, 4));
//
//	// Bottom
//	scenebuilder_add_mesh_instance(builder,
//		mesh_instance_new(1, ground,
//			transform_create(
//				vector3_create(0, -2, 8), vector3_create(16, 16, 16),
//				matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 2))));
//
//	Scene* scene = scene_new(builder,
//		camera_create(vector3_create(0, 2.5f, -1.5f),
//			matrix4_from_axis_angle(vector3_create(1, 0, 0), ILLUME_PI / 36), 90, 6.25, 0.05),
//		vector3_create(221.12f / 255.f, 248.45f / 255.f, 255.f / 255.f));
//	//vector3_create(0.001f, 0.001f, 0.001f));
//
//	scenebuilder_free(builder);
//	return scene;
//}

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
			//Scene* scene = init_scene();
			Scene* scene = scene_new("res/scenes/cornell_spheres/cornell_spheres.json");

			if (!scene)
			{
				goto exit_scene;
			}

			render_scene(scene, image, spp, max_depth);
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