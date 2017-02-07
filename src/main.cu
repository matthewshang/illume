#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "rapidjson/document.h"

#include "jsonutils.h"
#include "renderer.h"
#include "bitmap.h"
#include "scene/scene.h"

static char format[] = "%s-%dx%d-%dspp-%dmd.png";

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