#include "camera.h"

Camera camera_create(Vector3 pos, float fov, float dof, float aperture)
{
	Camera camera;
	camera.pos = pos;
	camera.fov = fov;
	camera.dof = dof;
	camera.aperture = aperture;
	return camera;
}