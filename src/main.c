#include <stdlib.h>
#include <stdio.h>

#include "kernel.h"
#include "bitmap.h"

int main(int argc, char* argv[])
{
	Bitmap* image = bitmap_new(960, 600);
	if (!image)
	{
		goto exit_bitmap;
	}

	// Scene* scene = scene_new(2);
	// if (!scene)
	// {
	// 	goto exit_scene;
	// }

	render_scene(image, 1024);

	if (argc > 1)
	{
		bitmap_save_to_png(image, argv[1]);
	}

	// scene_free(scene);

// exit_scene:
	bitmap_free(image);
exit_bitmap:
	return 0;
}
