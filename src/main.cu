#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "rapidjson/document.h"

#include "jsonutils.h"
#include "renderer.h"
#include "bitmap.h"
#include "math/mathutils.h"
#include "math/vector3.h"
#include "math/transform.h"
#include "math/matrix4.h"
#include "scene/scene.h"

static char format[] = "%s-%dx%d-%dspp-%dmd.png";

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
	if (argc < 5)
	{
		printf("illume: usage - <save> <scene> <spp> <maxdepth>\n");
		return 0;
	}
	
	int spp = strtol(argv[3], NULL, 10);
	int max_depth = strtol(argv[4], NULL, 10);

	rapidjson::Document d;
	JsonUtils::read_and_parse_json(argv[2], d);
	
	Scene* scene = scene_new(d);
	Renderer renderer(d, scene, spp, max_depth);
	Bitmap* image = bitmap_new(renderer.get_width(), renderer.get_height());	

	renderer.render_to_bitmap(image);
	
	char* name = (char *)calloc(1 + _snprintf(NULL, 0, format, 
		argv[1], renderer.get_width(), renderer.get_height(), spp, max_depth), sizeof(char));
	sprintf(name, format, argv[1], renderer.get_width(), renderer.get_height(), spp, max_depth);
	bitmap_save_to_png(image, name);
	printf("Saved to: %s\n", name);
	free(name);

	bitmap_free(image);
	scene_free(scene);
	
	return 0;
}