#ifndef _VECTOR3_
#define _VECTOR3_

typedef struct
{
	float x;
	float y;
	float z;
} 
Vector3;

#ifdef __cplusplus
extern "C" {
#endif

                     Vector3*  vector3_new            (float x, float y, float z);
                     void      vector3_free           (Vector3* vector);
__device__ __host__  Vector3   vector3_create         (float x, float y, float z);

__device__ __host__  void      vector3_set            (Vector3* v, float x, float y, float z);
__device__ __host__  void      vector3_normalize      (Vector3* v);
__device__ __host__  float     vector3_length2        (Vector3 v);
__device__ __host__  float     vector3_length         (Vector3 v);
__device__ __host__  float     vector3_dot            (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_cross          (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_to_basis       (Vector3 v, Vector3 normal);

__device__ __host__  Vector3   vector3_add            (Vector3 a, Vector3 b);
__device__ __host__  void      vector3_add_to         (Vector3* a, Vector3 b);
__device__ __host__  Vector3   vector3_sub            (Vector3 a, Vector3 b);
__device__ __host__  Vector3   vector3_mul            (Vector3 v, float m);
__device__ __host__  void      vector3_mul_vector_to  (Vector3* a, Vector3 b);


#ifdef __cplusplus
}
#endif

#endif