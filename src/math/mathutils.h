#ifndef _MATHUTILS_H_
#define _MATHUTILS_H_

#include <float.h>

#define ILLUME_PI 3.14159265358979323846
#define ILLUME_INV_PI 0.318309886183790671538

inline float degtorad(float deg)
{
	return deg * ILLUME_PI / 180.0f;
}

#endif