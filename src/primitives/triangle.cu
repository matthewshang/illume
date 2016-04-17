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
	tri.e10 = vector3_sub(v1, v0);
	tri.e20 = vector3_sub(v2, v0);
	tri.n = vector3_cross(tri.e10, tri.e20);
	vector3_normalize(&tri.n);
	return tri;
}