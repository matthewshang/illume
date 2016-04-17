#ifndef _TRIANGLE_
#define _TRIANGLE_

#include "../math/vector3.h"

typedef struct 
{
	Vector3 n;
	Vector3 v0;
	Vector3 e10;
	Vector3 e20;
}
Triangle;

Triangle  triangle_create  (Vector3 v0, Vector3 v1, Vector3 v2);

#endif