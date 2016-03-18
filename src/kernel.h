#ifndef _KERNEL_
#define _KERNEL_

#include "bitmap.h"

typedef struct
{
	float image_width;
	float image_height;
	float image_dim_ratio;
	float camera_tan_half_fov;
	float camera_focus_plane;
	float camera_width;
	float camera_height;
	float camera_left;
	float camera_top;
} 
RenderInfo;

static const float PI = 3.14159265358979323846;

#ifdef __cplusplus
extern "C" {
#endif

void call_kernel(Bitmap* bitmap);

#ifdef __cplusplus
}
#endif

#endif