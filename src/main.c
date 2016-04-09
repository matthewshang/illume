#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kernel.h"
#include "bitmap.h"

static char format[] = "%s-%sx%s-%sspp-%smd.png";

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
	{
		render_scene(image, spp, max_depth);
		char name[snprintf(NULL, 0, format, argv[1], argv[2], argv[3], argv[4], argv[5])];
		sprintf(name, format, argv[1], argv[2], argv[3], argv[4], argv[5]);
		printf("Saving to: %s\n", name);
		bitmap_save_to_png(image, name);
		bitmap_free(image);
	}
exit_bitmap:
	return 0;
}