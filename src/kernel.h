#ifndef _KERNEL_
#define _KERNEL_

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#include "bitmap.h"

typedef struct
{
	float image_width;
	float camera_focus_plane;
	float camera_pixel_size;
	float camera_left;
	float camera_top;
} 
RenderInfo;

static const float PI = 3.14159265358979323846;

#ifdef __cplusplus
extern "C" {
#endif

void render_scene(Bitmap* bitmap, int samples);

#ifdef __cplusplus
}
#endif

#endif