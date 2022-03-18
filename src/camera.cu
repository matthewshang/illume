#include "camera.h"

#include "jsonutils.h"
#include "math/transform.h"
#include "math/mathutils.h"

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

Camera camera_from_json(rapidjson::Value& json)
{
	Camera ret;
	JsonUtils::from_json(json, "fov",             ret.fov);
	JsonUtils::from_json(json, "depth_of_field",  ret.dof);
	JsonUtils::from_json(json, "aperture_radius", ret.aperture);
    auto transform = json.FindMember("transform");
    Transform t = transform_from_json(transform->value);
	ret.transform = t.mat;
	return ret;
}
