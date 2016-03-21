#include "vector3.h"

Vector3* vector3_new(float x, float y, float z)
{
	Vector3* vector = (Vector3 *) calloc(sizeof(Vector3), 1);
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
	if (v)
	{
		v->x = x;
		v->y = y;
		v->z = z;
	}
}

__device__ __host__
void vector3_normalize(Vector3* v)
{
	if (v)
	{
		float length = vector3_length(v);
		v->x /= length;
		v->y /= length;
		v->z /= length;
	}
}

__device__ __host__  
float vector3_length2(Vector3* v)
{
	return (v->x * v->x) + (v->y * v->y) + (v->z * v->z);
}

__device__ __host__  
float vector3_length(Vector3* v)
{
	if (v)
	{
		return sqrtf(vector3_length2(v));
	}

	return 0;
}

__device__ __host__
Vector3 vector3_cross(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		return vector3_create(a->y * b->z - a->z * b->y,
							  a->z * b->x - a->x * b->z,
							  a->x * b->y - a->y * b->x);
	}

	return vector3_create(0, 0, 0);
}

__device__ __host__
float vector3_dot(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		return (a->x * b->x) + (a->y * b->y) + (a->z * b->z);
	}

	return 0;
}

__device__ __host__
Vector3 vector3_to_basis(Vector3* v, Vector3* normal)
{
	if (v && normal)
	{
		Vector3 tangent;
		if (normal->x == 0)
		{
			vector3_set(&tangent, 1, 0, 0);
		}
		else
		{
			vector3_set(&tangent, 0, -1 * normal->z, normal->y);
		}
		vector3_normalize(&tangent);
		Vector3 bittangent = vector3_cross(&tangent, normal);
		Vector3 x = vector3_mul(&tangent, v->x);
		Vector3 y = vector3_mul(normal, v->y);
		Vector3 z = vector3_mul(&bittangent, v->z);
		vector3_add_to(&y, &z);
		return vector3_add(&x, &y);
	}

	return vector3_create(0, 0, 0);
}

__device__ __host__
Vector3 vector3_add(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		return vector3_create(a->x + b->x, a->y + b->y, a->z + b->z);		
	}

	return vector3_create(0, 0, 0);
}

__device__ __host__
void vector3_add_to(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		a->x += b->x; 
		a->y += b->y;
		a->z += b->z;		
	}
}

__device__ __host__
Vector3 vector3_sub(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		return vector3_create(a->x - b->x, a->y - b->y, a->z - b->z);		
	}

	return vector3_create(0, 0, 0);
}

__device__ __host__
Vector3 vector3_mul(Vector3* v, float m)
{
	if (v)
	{
		return vector3_create(v->x * m, v->y * m, v->z * m);
	}
	
	return vector3_create(0, 0, 0);
}

__device__ __host__
Vector3 vector3_mul_vector(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		return vector3_create(a->x * b->x, a->y * b->y, a->z * b->z);
	}

	return vector3_create(0, 0, 0);
}

__device__ __host__
void vector3_mul_vector_to(Vector3* a, Vector3* b)
{
	if (a && b)
	{
		vector3_set(a, a->x * b->x, a->y * b->y, a->z * b->z);
	}
}