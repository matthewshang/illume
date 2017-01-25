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
	JsonUtils::from_json(json, "transform",       ret.pos);
	JsonUtils::from_json(json, "fov",             ret.fov);
	JsonUtils::from_json(json, "depth_of_field",  ret.dof);
	JsonUtils::from_json(json, "aperture_radius", ret.aperture);
	auto transform = json.FindMember("transform");
	Vector3 translation, rotation;
	if (transform != json.MemberEnd())
	{
		JsonUtils::from_json(transform->value, "translation", translation, vector3_create(0, 0, 0));
		JsonUtils::from_json(transform->value, "rotation", rotation, vector3_create(0, 0, 0));
	}
	ret.transform = matrix4_create();
	matrix4_set_translate(&ret.transform, translation);
	Matrix4 x = matrix4_from_axis_angle(vector3_create(1, 0, 0), degtorad(rotation.x));
	Matrix4 y = matrix4_from_axis_angle(vector3_create(0, 1, 0), degtorad(rotation.y));
	Matrix4 z = matrix4_from_axis_angle(vector3_create(0, 0, 1), degtorad(rotation.z));
	ret.transform = matrix4_mul(ret.transform, matrix4_mul(matrix4_mul(x, y), z));
	return ret;
}
