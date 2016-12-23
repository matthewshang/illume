#ifndef _CAMERA_
#define _CAMERA_

#include "rapidjson/document.h"

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

Camera  camera_create     (Vector3 pos, Matrix4 rotation, float fov, float dof, float aperture);
Camera  camera_from_json  (rapidjson::Value& json);

#ifdef __cplusplus
}
#endif

#endif