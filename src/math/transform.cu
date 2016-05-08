#include "transform.h"

Transform transform_create(Vector3 translation, Vector3 scale)
{
	Transform transform;
	transform.mat = matrix4_create();
	matrix4_set_translate(&transform.mat, translation);
	matrix4_set_scale(&transform.mat, scale);
	transform.inv = matrix4_get_inverse(transform.mat);
	transform.trans = matrix4_get_transpose(transform.mat);
	transform.trans_inv = matrix4_get_transpose(transform.inv);
	return transform;
}