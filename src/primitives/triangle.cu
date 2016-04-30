#include "triangle.h"

Triangle* triangle_new(Vector3 v0, Vector3 v1, Vector3 v2)
{
	Triangle* tri = (Triangle *) calloc(1, sizeof(Triangle));
	if (!tri)
	{
		return NULL;
	}
	*tri = triangle_create(v0, v1, v2);
	return tri;
}

void triangle_free(Triangle* triangle)
{
	if (triangle)
	{
		free(triangle);
	}
}

Triangle triangle_create(Vector3 v0, Vector3 v1, Vector3 v2)
{
	Triangle tri;
	tri.v0 = v0;
	Vector3 e10 = vector3_sub(v1, v0);
	Vector3 e20 = vector3_sub(v2, v0);
	tri.n = vector3_cross(e10, e20);
	vector3_normalize(&tri.n);

	tri.ex = e10;
	vector3_normalize(&tri.ex);
	tri.ey = vector3_cross(tri.ex, tri.n);
	tri.t1x = vector3_dot(tri.ex, e10);
	tri.t1y = vector3_dot(tri.ey, e10);
	tri.t2x = vector3_dot(tri.ex, e20);
	tri.t2y = vector3_dot(tri.ey, e20);
	tri.area = 1e-4 + tri_area_times_two(0, 0, tri.t1x, tri.t1y, tri.t2x, tri.t2y);
	return tri;
}

__device__ __host__
float tri_area_times_two(float ax, float ay, float bx, float by, float cx, float cy)
{
	return fabsf(ax * by + bx * cy + cx * ay - ay * bx - by * cx - cy * ax);
}