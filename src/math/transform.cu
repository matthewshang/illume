#include "transform.h"

#include "mathutils.h"
#include "../jsonutils.h"

Transform transform_create(Vector3 translation, Vector3 scale, Matrix4 rotation)
{
	Transform transform;
	Matrix4 translate = matrix4_create();
	matrix4_set_translate(&translate, translation);
	transform.mat = matrix4_mul(matrix4_mul(translate, rotation), matrix4_from_scale(scale));
	transform.inv = matrix4_get_inverse(transform.mat);
	transform.trans = matrix4_get_transpose(transform.mat);
	transform.trans_inv = matrix4_get_transpose(transform.inv);
	return transform;
}

Transform transform_from_matrix(Matrix4& mat)
{
    Transform transform;
    transform.mat = mat;
    transform.inv = matrix4_get_inverse(transform.mat);
    transform.trans = matrix4_get_transpose(transform.mat);
    transform.trans_inv = matrix4_get_transpose(transform.inv);
    return transform;
}

Transform transform_from_json(rapidjson::Value& json)
{
	Vector3 translation, scale, rotation;
	JsonUtils::from_json(json, "translation", translation, vector3_create(0, 0, 0));
	JsonUtils::from_json(json, "scale",       scale, vector3_create(1, 1, 1));
	JsonUtils::from_json(json, "rotation",    rotation, vector3_create(0, 0, 0));
    if (json.HasMember("lookat"))
    {
        Vector3 lookat;
        JsonUtils::from_json(json, "lookat", lookat);
        return transform_from_matrix(matrix4_lookat(translation, vector3_sub(lookat, translation), vector3_create(0, 1, 0)));
    }
    Matrix4 x = matrix4_from_axis_angle(vector3_create(1, 0, 0), degtorad(rotation.x));
    Matrix4 y = matrix4_from_axis_angle(vector3_create(0, 1, 0), degtorad(rotation.y));
    Matrix4 z = matrix4_from_axis_angle(vector3_create(0, 0, 1), degtorad(rotation.z));
    return transform_create(translation, scale, matrix4_mul(matrix4_mul(x, y), z));
}
