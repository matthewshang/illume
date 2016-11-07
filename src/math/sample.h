#ifndef _SAMPLE_
#define _SAMPLE_

#include "vector3.h"

#ifdef __cplusplus
extern "C" {
#endif

__device__  Vector3  sample_hemisphere_cosine  (float u1, float u2);
__device__  Vector3  sample_circle             (float u1, float u2);
__device__  Vector3  sample_sphere             (float u1, float u2);

#ifdef __cplusplus
}
#endif

#endif