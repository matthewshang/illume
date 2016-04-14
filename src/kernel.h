#ifndef _KERNEL_
#define _KERNEL_

#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include <curand.h>
#include <curand_kernel.h>

#include "bitmap.h"
#include "vector3.h"
#include "ray.h"
#include "sphere.h"
#include "sample.h"
#include "scene.h"
#include "material.h"
#include "error_check.h"
#include "plane.h"
#include "camera.h"
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

void  render_scene  (Scene* scene, Bitmap* bitmap, Camera camera, int samples, int max_depth);

#ifdef __cplusplus
}
#endif

#endif