#include "transform.h"

Transform transform_create(Vector3 translation, Vector3 scale, Matrix4 rotation)
{
	Transform transform;
	Matrix4 translate = matrix4_create();
	matrix4_set_translate(&translate, translation);
	transform.mat = matrix4_mul(matrix4_mul(translate, matrix4_from_scale(scale)), rotation);
	transform.inv = matrix4_get_inverse(transform.mat);
	transform.trans = matrix4_get_transpose(transform.mat);
	transform.trans_inv = matrix4_get_transpose(transform.inv);
	return transform;
}