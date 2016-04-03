#include <stdlib.h>
#include <stdio.h>

#include "kernel.h"
#include "bitmap.h"

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		printf("illume: usage - <path> <width> <height> <spp>\n");
		goto exit_bitmap;
	}

	int width = atoi(argv[2]);
	int height = atoi(argv[3]);
	int spp = atoi(argv[4]);

	Bitmap* image = bitmap_new(width, height);
	if (!image)
	{
		goto exit_bitmap;
	}

	// Scene* scene = scene_new(2);
	// if (!scene)
	// {
	// 	goto exit_scene;
	// }

	render_scene(image, spp);
	bitmap_save_to_png(image, argv[1]);

	// scene_free(scene);

// exit_scene:
	bitmap_free(image);
exit_bitmap:
	return 0;
}
