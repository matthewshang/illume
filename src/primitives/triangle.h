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

#ifdef __cplusplus
extern "C" {
#endif

Triangle*  triangle_new     (Vector3 v0, Vector3 v1, Vector3 v2);
void       triangle_free    (Triangle* triangle);
Triangle   triangle_create  (Vector3 v0, Vector3 v1, Vector3 v2);

#ifdef __cplusplus
}
#endif

#endif