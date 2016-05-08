#include "transform.h"

Transform transform_create(Vector3 translation, Vector3 scale, Matrix4 rotation)
{
	Transform transform;
	transform.mat = matrix4_mul(matrix4_from_scale(scale), rotation);
	matrix4_set_translate(&transform.mat, translation);
	transform.inv = matrix4_get_inverse(transform.mat);
	transform.trans = matrix4_get_transpose(transform.mat);
	transform.trans_inv = matrix4_get_transpose(transform.inv);
	return transform;
}