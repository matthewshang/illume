#ifndef _TRANSFORM_
#define _TRANSFORM_

#include "matrix4.h"
#include "vector3.h"

typedef struct
{
	Matrix4 mat;
	Matrix4 inv;
	Matrix4 trans;
	Matrix4 trans_inv;
}
Transform;

#ifdef __cplusplus
extern "C" {
#endif

Transform  transform_create  (Vector3 translation, Vector3 scale);

#ifdef __cplusplus
}
#endif

#endif