#ifndef _MATRIX4_
#define _MATRIX4_

#include "vector3.h"

const int SUB[4][3] = 
{
	{1, 2, 3},	
    {0, 2, 3},
    {0, 1, 3},
    {0, 1, 2}
};

typedef struct
{
	float m[4][4];
}
Matrix4;

#ifdef __cplusplus
extern "C" {
#endif
			
                      Matrix4  matrix4_create           ();
                      Matrix4  matrix4_from_axis_angle  (Vector3 axis, float angle);
                      Matrix4  matrix4_from_scale       (Vector3 scale);
                      void     matrix4_set_translate    (Matrix4* m, Vector3 translation);
                      Matrix4  matrix4_mul              (Matrix4 a, Matrix4 b);
                      Matrix4  matrix4_get_transpose    (Matrix4 m);
                      Matrix4  matrix4_get_inverse      (Matrix4 m);
__device__  __host__  Vector3  matrix4_mul_vector3      (Matrix4* m, Vector3 v, float w);

#ifdef __cplusplus
}
#endif

#endif