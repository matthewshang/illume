#include "vector3.h"

#include <math.h>

Vector3* vector3_new(float x, float y, float z)
{
	Vector3* vector = (Vector3 *) calloc(1, sizeof(Vector3));
	if (!vector)
	{
		return NULL;
	}

	vector->x = x;
	vector->y = y;
	vector->z = z;
	return vector;
}

void vector3_free(Vector3* vector)
{
	if (vector)
	{
		free(vector);
	}
}

__device__ __host__
Vector3 vector3_create(float x, float y, float z)
{
	Vector3 vector;
	vector3_set(&vector, x, y, z);
	return vector;
}

__device__ __host__
void vector3_set(Vector3* v, float x, float y, float z)
{
	v->x = x;
	v->y = y;
	v->z = z;
}

__device__ __host__
void vector3_normalize(Vector3* v)
{

	float invLength = 1.0f / vector3_length(*v);
	v->x *= invLength;
    v->y *= invLength;
	v->z *= invLength;
}

__device__ __host__ Vector3 vector3_normalized(const Vector3& v)
{
    float invLen = 1.0f / vector3_length(v);
    return vector3_create(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __host__  
float vector3_length2(Vector3 v)
{
	return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
}

__device__ __host__  
float vector3_length(Vector3 v)
{
	return sqrtf(vector3_length2(v));
}

__device__ __host__
Vector3 vector3_cross(Vector3 a, Vector3 b)
{
	return vector3_create(a.y * b.z - a.z * b.y,
						  a.z * b.x - a.x * b.z,
						  a.x * b.y - a.y * b.x);
}

__device__ __host__
float vector3_dot(Vector3 a, Vector3 b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__device__ __host__
Vector3 vector3_to_basis(Vector3 v, Vector3 normal)
{
	Vector3 tangent;
	if (fabsf(normal.x) > fabsf(normal.y))
	{
		tangent = vector3_create(normal.z, 0.0f, -normal.x);
	}
	else
	{
		tangent = vector3_create(0.0f, normal.z, -normal.y);
	}
	Vector3 bitangent = vector3_cross(normal, tangent);
	tangent = vector3_cross(bitangent, normal);
	Vector3 b = vector3_add(vector3_add(vector3_mul(tangent, v.x), 
								   vector3_mul(normal, v.y)), 
								   vector3_mul(bitangent, v.z));
	vector3_normalize(&b);
	return b;
}

__device__ __host__ 
Vector3 vector3_min(Vector3 v, float m)
{
	return vector3_create(fminf(v.x, m), fminf(v.y, m), fminf(v.z, m));
}

__device__ __host__ 
Vector3 vector3_max(Vector3 v, float m)
{
	return vector3_create(fmaxf(v.x, m), fmaxf(v.y, m), fmaxf(v.z, m));
}

__device__ __host__
Vector3 vector3_pow(Vector3 v, float p)
{
	return vector3_create(powf(v.x, p), powf(v.y, p), powf(v.z, p));
}

__device__ __host__ Vector3 vector3_exp(const Vector3 & v)
{
    return vector3_create(expf(v.x), expf(v.y), expf(v.z));
}

__device__ __host__
Vector3 vector3_reflect(Vector3 v, Vector3 n)
{
	return vector3_sub(v, vector3_mul(vector3_mul(n, vector3_dot(v, n)), 2));
}

__device__ __host__
Vector3 vector3_add(Vector3 a, Vector3 b)
{
	return vector3_create(a.x + b.x, a.y + b.y, a.z + b.z);		
}

__device__ __host__ Vector3 vector3_add(const Vector3& a, float value)
{
    return vector3_create(value + a.x, value + a.y, value + a.z);
}

__device__ __host__
void vector3_add_to(Vector3* a, Vector3 b)
{
	a->x += b.x; 
	a->y += b.y;
	a->z += b.z;		
}

__device__ __host__
Vector3 vector3_sub(Vector3 a, Vector3 b)
{
	return vector3_create(a.x - b.x, a.y - b.y, a.z - b.z);		
}

__device__ __host__
Vector3 vector3_mul(Vector3 v, float m)
{
	return vector3_create(v.x * m, v.y * m, v.z * m);
}

__device__ __host__
Vector3 vector3_mul(const Vector3& a, const Vector3& b)
{
	return vector3_create(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__
void vector3_mul_vector_to(Vector3* a, Vector3 b)
{
	vector3_set(a, a->x * b.x, a->y * b.y, a->z * b.z);
}

__device__ __host__ Vector3 vector3_div(const Vector3& a, const Vector3& b)
{
    return vector3_create(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __host__ Vector3 vector3_div(const Vector3& a, float s)
{
    return vector3_create(a.x / s, a.y / s, a.z / s);
}
