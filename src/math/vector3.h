#ifndef _VECTOR3_
#define _VECTOR3_

#include "../intellisense.h"

struct Vector3
{
	float x;
	float y;
	float z;
};


                     Vector3*  vector3_new            (float x, float y, float z);
                     void      vector3_free           (Vector3* vector);
__device__ __host__  Vector3   vector3_create         (float x, float y, float z);

__device__ __host__  void      vector3_set            (Vector3* v, float x, float y, float z);
__device__ __host__  void      vector3_normalize      (Vector3* v);
__device__ __host__  Vector3   vector3_normalized     (const Vector3& v);
__device__ __host__  float     vector3_length2        (Vector3 v);
__device__ __host__  float     vector3_length         (Vector3 v);
__device__ __host__  float     vector3_dot            (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_cross          (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_to_basis       (Vector3 v, Vector3 normal);

__device__ __host__  Vector3   vector3_min            (Vector3 v, float m);
__device__ __host__  Vector3   vector3_max            (Vector3 v, float m);
__device__ __host__  Vector3   vector3_pow            (Vector3 v, float p);
__device__ __host__  Vector3   vector3_exp            (const Vector3& v);
__device__ __host__  Vector3   vector3_reflect        (Vector3 v, Vector3 n);

__device__ __host__  Vector3   vector3_add            (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_add            (const Vector3& a, float value);
__device__ __host__  void      vector3_add_to         (Vector3* a, Vector3 b);
__device__ __host__  Vector3   vector3_sub            (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_mul            (Vector3 v, float m);
__device__ __host__  Vector3   vector3_mul            (const Vector3& a, const Vector3& b);
__device__ __host__  void      vector3_mul_vector_to  (Vector3* a, Vector3 b);
__device__ __host__  Vector3   vector3_div            (const Vector3& a, const Vector3& b);
__device__ __host__  Vector3   vector3_div            (const Vector3& a, float s);

#endif