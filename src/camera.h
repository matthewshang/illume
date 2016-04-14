#ifndef _CAMERA_
#define _CAMERA_

#include "vector3.h"

typedef struct
{
	Vector3 pos;
	float fov;
	float dof;
	float aperture;
}
Camera;

#ifdef __cplusplus
extern "C" {
#endif

Camera  camera_create  (Vector3 pos, float fov, float dof, float aperture);

#ifdef __cplusplus
}
#endif

#endif