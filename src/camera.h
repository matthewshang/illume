#ifndef _CAMERA_
#define _CAMERA_

#include "math/vector3.h"
#include "math/matrix4.h"

typedef struct
{
	Vector3 pos;
	float fov;
	float dof;
	float aperture;
	Matrix4 transform;
}
Camera;

#ifdef __cplusplus
extern "C" {
#endif

Camera  camera_create  (Vector3 pos, Matrix4 rotation, float fov, float dof, float aperture);

#ifdef __cplusplus
}
#endif

#endif