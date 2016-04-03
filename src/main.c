#include <stdlib.h>
#include <stdio.h>

#include "kernel.h"
#include "bitmap.h"

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		printf("illume: usage - <path> <width> <height> <spp> <maxdepth>\n");
		goto exit_bitmap;
	}

	int width = atoi(argv[2]);
	int height = atoi(argv[3]);
	int spp = atoi(argv[4]);
	int max_depth = atoi(argv[5]);

	Bitmap* image = bitmap_new(width, height);
	if (!image)
	{
		goto exit_bitmap;
	}

	render_scene(image, spp, max_depth);
	bitmap_save_to_png(image, argv[1]);
	bitmap_free(image);
exit_bitmap:
	return 0;
}
