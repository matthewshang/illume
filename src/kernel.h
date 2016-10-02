#ifndef _KERNEL_
#define _KERNEL_

#include "bitmap.h"
#include "scene/scene.h"
#include "camera.h"

#ifdef __cplusplus
extern "C" {
#endif

void  render_scene  (Scene* scene, Bitmap* bitmap, Camera camera, int samples, int max_depth);

#ifdef __cplusplus
}
#endif

#endif