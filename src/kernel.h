#ifndef _KERNEL_
#define _KERNEL_

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#include "bitmap.h"

static const float PI = 3.14159265358979323846;

#ifdef __cplusplus
extern "C" {
#endif

void render_scene(Bitmap* bitmap, int samples, int max_depth);

#ifdef __cplusplus
}
#endif

#endif