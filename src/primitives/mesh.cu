#include "mesh.h"

Mesh* mesh_new(Material m)
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;
	mesh->m = m;
	return mesh;
}


void mesh_free(Mesh* mesh)
{
	if (mesh)
	{
		if (mesh->triangles)
		{
			free(mesh->triangles);
		}
		free(mesh);
	}
}

static void obj_get_token(char* line, char** ret)
{
	if (!line) 
	{
		return;
	}
	char* dup = strdup(line);
	char* p = strtok(dup, " ");
	if (p)
	{
		*ret = strdup(p);
	}
	free(dup);
}

static void split_string(const char* string, const char* delim, int tokens, char** ret)
{
	char* dup = strdup(string);
	char* p = strtok(dup, delim);
	int i = 0;
	while (i < tokens && p != NULL)
	{
		ret[i++] = strdup(p);
		p = strtok(NULL, delim);
	}

	free(dup);
}

static void split_string_finish(char** tokens, int amount)
{
	for (int i = 0; i < amount; i++)
	{
		if (tokens[i]) 
		{	
			free(tokens[i]);
		}
	}
}

void mesh_load_obj(Mesh* mesh, const char* path)
{
	FILE* file;
	file = fopen(path, "rt");
	if (!file)
	{
		printf("mesh_load_obj: cannot read file %s\n", path);
		return;
	}

	char line[100];
	char* token;
	char** tokens = (char **) calloc(OBJ_TOKENS, sizeof(char *));
	ArrayList* vertices = arraylist_new(3);
	ArrayList* triangles = arraylist_new(1);

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			arraylist_add(vertices, vector3_new(strtof(tokens[1], NULL),
												strtof(tokens[2], NULL) - 0.15,
											    strtof(tokens[3], NULL) + 5));
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
		else if (strcmp(token, TOKEN_FACE) == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
			int i0 = strtol(tokens[1], NULL, 10) - 1;
			int i1 = strtol(tokens[2], NULL, 10) - 1;
			int i2 = strtol(tokens[3], NULL, 10) - 1;
			Vector3* v0 = (Vector3 *) arraylist_get(vertices, i0);
			Vector3* v1 = (Vector3 *) arraylist_get(vertices, i1);
			Vector3* v2 = (Vector3 *) arraylist_get(vertices, i2);

			arraylist_add(triangles, triangle_new(*v0, *v1, *v2));
			split_string_finish(tokens, FACE_COMPONENTS);
		}

		if (token) 
		{
			free(token);
		}
	}
	mesh->triangles = (Triangle *) calloc(triangles->length, sizeof(Triangle));
	if (mesh->triangles)
	{
		for (int i = 0; i < triangles->length; i++)
		{
			mesh->triangles[i] = *((Triangle *) arraylist_get(triangles, i));
		}
		mesh->triangle_amount = triangles->length;
	}
	else
	{
		printf("mesh_load_obj: allocation of mesh tris failed");
	}

	for (int i = 0; i < vertices->length; i++)
	{
		vector3_free((Vector3 *) arraylist_get(vertices, i));
	}
	arraylist_free(vertices);
	for (int i = 0; i < triangles->length; i++)
	{
		triangle_free((Triangle *) arraylist_get(triangles, i));
	}
	arraylist_free(triangles);
	fclose(file);
	if (tokens)
	{ 
		free(tokens);
	}
}

__device__
static float tri_area_times_two(float ax, float ay, float bx, float by, float cx, float cy)
{
	return fabsf(ax * by + bx * cy + cx * ay - ay * bx - by * cx - cy * ax);
}

__device__
static int point_in_triangle(float bx, float by, float cx, float cy, float px, float py)
{
	float ABC = tri_area_times_two(0, 0, bx, by, cx, cy);
	float ABP = tri_area_times_two(0, 0, bx, by, px, py);
	float BCP = tri_area_times_two(bx, by, cx, cy, px, py);
	float CAP = tri_area_times_two(cx, cy, 0, 0, px, py);
	return (ABP + BCP + CAP < ABC + 1e-4);
}

__device__
static float triangle_ray_intersect(Triangle tri, Ray ray)
{
	float d = vector3_dot(tri.n, vector3_sub(tri.v0, ray.o)) / vector3_dot(tri.n, ray.d);
	if (d < 0)
	{
		return -1;
	}
	Vector3 point = ray_position_along(ray, d);
	Vector3 p0 = vector3_sub(point, tri.v0);
	Vector3 ex = tri.e10;
	vector3_normalize(&ex);
	Vector3 ey = vector3_cross(ex, tri.n);

	// one point of triangle is at origin 
	if (point_in_triangle(vector3_dot(tri.e10, ex), vector3_dot(tri.e10, ey),
						  vector3_dot(tri.e20, ex), vector3_dot(tri.e20, ey),
						  vector3_dot(p0, ex), vector3_dot(p0, ey)))
	{
		return d;
	}

	return -1;
}

__device__
Intersection mesh_ray_intersect(Mesh mesh, Ray ray)
{
	Intersection min = intersection_create_no_intersect();
	min.d = FLT_MAX;
	for (int i = 0; i < mesh.triangle_amount; i++)
	{
		Triangle tri = mesh.triangles[i];
		float inter_d = triangle_ray_intersect(tri, ray);

		if (inter_d > 0 && inter_d < min.d)
		{
			min.d = inter_d;
			min.normal = tri.n;
			min.is_intersect = 1;
		}
	}
	if (min.is_intersect)
	{
		min.m = mesh.m;
	}

	return min;
}