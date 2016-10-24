#include "camera.h"

Camera camera_create(Vector3 pos, Matrix4 rotation, float fov, float dof, float aperture)
{
	Camera camera;
	camera.pos = pos;
	camera.fov = fov;
	camera.dof = dof;
	camera.aperture = aperture;
	camera.transform = matrix4_create();
	matrix4_set_translate(&camera.transform, pos);
	camera.transform = matrix4_mul(camera.transform, rotation);
	return camera;
}