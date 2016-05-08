#include "matrix4.h"

void matrix4_print(Matrix4 m)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("%f    ", m.m[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

Matrix4 matrix4_create()
{
	Matrix4 m;
	m.m[0][0] = 1;  m.m[0][1] = 0;  m.m[0][2] = 0;  m.m[0][3] = 0;
	m.m[1][0] = 0;  m.m[1][1] = 1;  m.m[1][2] = 0;  m.m[1][3] = 0;
	m.m[2][0] = 0;  m.m[2][1] = 0;  m.m[2][2] = 1;  m.m[2][3] = 0;
	m.m[3][0] = 0;  m.m[3][1] = 0;  m.m[3][2] = 0;  m.m[3][3] = 1;
	return m;
}

Matrix4 matrix4_from_axis_angle(Vector3 axis, float angle)
{
	Vector3 v = axis;
	vector3_normalize(&axis);
	v = vector3_mul(axis, sinf(angle / 2));
	float w = cosf(angle / 2);

	Matrix4 m = matrix4_create();
	m.m[0][0] = 1 - 2 * (v.y * v.y + v.z * v.z);
	m.m[1][0] = 2 * (v.x * v.y + w * v.z);
	m.m[2][0] = 2 * (v.x * v.z - w * v.y);
	m.m[0][1] = 2 * (v.x * v.y - w * v.z);
	m.m[1][1] = 1 - 2 * (v.x * v.x + v.z * v.z);
	m.m[2][1] = 2 * (v.y * v.z + w * v.x);
	m.m[0][2] = 2 * (v.x * v.z + w * v.y);
	m.m[1][2] = 2 * (v.y * v.z - w * v.x);
	m.m[2][2] = 1 - 2 * (v.x * v.x + v.y * v.y);
	return m;
}

Matrix4 matrix4_from_scale(Vector3 scale)
{
	Matrix4 m = matrix4_create();
	m.m[0][0] = scale.x;
	m.m[1][1] = scale.y;
	m.m[2][2] = scale.z;
	return m;
}

void matrix4_set_translate(Matrix4* m, Vector3 translation)
{
	if (m)
	{
		m->m[0][3] = translation.x;
		m->m[1][3] = translation.y;
		m->m[2][3] = translation.z;
	}
}

Matrix4 matrix4_mul(Matrix4 a, Matrix4 b)
{
	Matrix4 m;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m.m[i][j] = a.m[i][0] * b.m[0][j] +
			            a.m[i][1] * b.m[1][j] +
			            a.m[i][2] * b.m[2][j] +
			            a.m[i][3] * b.m[3][j];
		}
	}
	return m;
}

Matrix4 matrix4_get_transpose(Matrix4 m)
{
	Matrix4 t;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			t.m[j][i] = m.m[i][j];
		}
	}
	return t;
}

static void mul_float_to(Matrix4* m, float x);
static Matrix4 adjoint(Matrix4 m);
static float det(Matrix4 m);

Matrix4 matrix4_get_inverse(Matrix4 m)
{
	Matrix4 in = adjoint(m);
	mul_float_to(&in, 1 / det(m));
	return in;
}

static float subdet(Matrix4 m, int i, int j)
{
	float d[3][3];
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			d[y][x] = m.m[SUB[i][y]][SUB[j][x]];
		}
	}
	return (d[0][0] * d[1][1] * d[2][2]) 
	     + (d[0][1] * d[1][2] * d[2][0]) 
	     + (d[0][2] * d[1][0] * d[2][1])
		 - (d[0][2] * d[1][1] * d[2][0]) 
		 - (d[0][1] * d[1][0] * d[2][2]) 
		 - (d[0][0] * d[1][2] * d[2][1]); 
}

static Matrix4 adjoint(Matrix4 m)
{
	Matrix4 ad;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if ((i + j) % 2 == 0)
			{
				ad.m[i][j] = subdet(m, j, i);
			}
			else
			{
				ad.m[i][j] = -1 * subdet(m, j, i);
			}
		}
	}
	return ad;
}

static void mul_float_to(Matrix4* m, float x)
{
	if (m)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m->m[i][j] *= x;
			}
		}
	}
}

static float det(Matrix4 m)
{
	return (m.m[0][0] * m.m[1][1] * m.m[2][2] * m.m[3][3]) 
	     + (m.m[0][0] * m.m[1][2] * m.m[2][3] * m.m[3][1]) 
	     + (m.m[0][0] * m.m[1][3] * m.m[2][1] * m.m[3][2]) 
	     + (m.m[0][1] * m.m[1][0] * m.m[2][3] * m.m[3][2]) 
	     + (m.m[0][1] * m.m[1][2] * m.m[2][0] * m.m[3][3]) 
	     + (m.m[0][1] * m.m[1][3] * m.m[2][2] * m.m[3][0])
		 + (m.m[0][2] * m.m[1][0] * m.m[2][1] * m.m[3][3]) 
		 + (m.m[0][2] * m.m[1][1] * m.m[2][3] * m.m[3][0]) 
		 + (m.m[0][2] * m.m[1][3] * m.m[2][0] * m.m[3][1]) 
		 + (m.m[0][3] * m.m[2][0] * m.m[2][2] * m.m[3][1]) 
		 + (m.m[0][3] * m.m[1][1] * m.m[2][0] * m.m[3][2]) 
		 + (m.m[0][3] * m.m[1][2] * m.m[2][1] * m.m[3][0])

		 - (m.m[0][0] * m.m[1][1] * m.m[2][3] * m.m[3][2]) 
		 - (m.m[0][0] * m.m[1][2] * m.m[2][1] * m.m[3][3]) 
		 - (m.m[0][0] * m.m[1][3] * m.m[2][2] * m.m[3][1]) 
		 - (m.m[0][1] * m.m[1][0] * m.m[2][2] * m.m[3][3]) 
		 - (m.m[0][1] * m.m[1][2] * m.m[2][3] * m.m[3][0]) 
		 - (m.m[0][1] * m.m[1][3] * m.m[2][0] * m.m[3][2])
		 - (m.m[0][2] * m.m[1][0] * m.m[2][3] * m.m[3][1]) 
		 - (m.m[0][2] * m.m[1][1] * m.m[2][0] * m.m[3][3]) 
		 - (m.m[0][2] * m.m[1][3] * m.m[2][1] * m.m[3][0]) 
		 - (m.m[0][3] * m.m[1][0] * m.m[2][1] * m.m[3][2]) 
		 - (m.m[0][3] * m.m[1][1] * m.m[2][2] * m.m[3][0]) 
		 - (m.m[0][3] * m.m[1][2] * m.m[2][0] * m.m[3][1]);
}

__device__
Vector3 matrix4_mul_vector3(Matrix4* m, Vector3 v, float w)
{
	return vector3_create(m->m[0][0] * v.x + m->m[0][1] * v.y + m->m[0][2] * v.z + m->m[0][3] * w, 
						  m->m[1][0] * v.x + m->m[1][1] * v.y + m->m[1][2] * v.z + m->m[1][3] * w,
						  m->m[2][0] * v.x + m->m[2][1] * v.y + m->m[2][2] * v.z + m->m[2][3] * w);
}
