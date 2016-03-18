#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "kernel.h"
#include "bitmap.h"

int main(int argc, char** argv)
{
	Bitmap* image = bitmap_new(512, 512);
	if (!image)
	{
		goto exit;
	}

	struct timespec tstart = {0, 0};
	struct timespec tend = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	call_kernel(image);

	clock_gettime(CLOCK_MONOTONIC, &tend);
	printf("%f seconds\n", 
		    ((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
		    ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

	if (argc > 1)
	{
		bitmap_save_to_png(image, argv[1]);
	}

	bitmap_free(image);
	
exit:
	return 0;
}
