  #include "mesh.h"

#include <stdio.h>
#include <string.h>

#include "../arraylist.h"
#include "../math/constants.h"

typedef struct
{
	ArrayList* vertices;
	ArrayList* triangles;
	ArrayList* aabbs;
}
tmp_mesh;

__device__ __host__
static float tri_area_times_two(float ax, float ay, float bx, float by, float cx, float cy)
{
	return fabsf(ax * by + bx * cy + cx * ay - ay * bx - by * cx - cy * ax);
}

static Triangle triangle_create(Vector3 v0, Vector3 v1, Vector3 v2)
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

static Triangle* triangle_new(Vector3 v0, Vector3 v1, Vector3 v2)
{
	Triangle* tri = (Triangle *) calloc(1, sizeof(Triangle));
	if (!tri)
	{
		return NULL;
	}
	*tri = triangle_create(v0, v1, v2);
	return tri;
}

static void triangle_free(Triangle* triangle)
{
	if (triangle)
	{
		free(triangle);
	}
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

static void fix_aabb(AABB* aabb)
{
	if (aabb->max.x - aabb->min.x < FLT_EPSILON)
	{
		aabb->max.x += FLT_EPSILON;
	}
	if (aabb->max.y - aabb->min.y < FLT_EPSILON)
	{
		aabb->max.y += FLT_EPSILON;
	}
	if (aabb->max.z - aabb->min.z < FLT_EPSILON)
	{
		aabb->max.z += FLT_EPSILON;
	}
}

static void load_obj(Mesh* mesh, const char* path, tmp_mesh* tmp)
{
	FILE* file;
	file = fopen(path, "rt");
	if (!file)
	{
		printf("mesh_load_obj: cannot read file %s\n", path);
		return;
	}
	printf("Loading mesh from %s...\n", path);

	char line[100];
	char* token;
	char** tokens = (char **) calloc(OBJ_TOKENS, sizeof(char *));
	tmp->vertices = arraylist_new(3);
	tmp->triangles = arraylist_new(1);
	tmp->aabbs = arraylist_new(1);

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			arraylist_add(tmp->vertices, vector3_new(strtof(tokens[1], NULL),
												strtof(tokens[2], NULL),
											    strtof(tokens[3], NULL)));
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
		else if (strcmp(token, TOKEN_FACE) == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
			int i0 = strtol(tokens[1], NULL, 10) - 1;
			int i1 = strtol(tokens[2], NULL, 10) - 1;
			int i2 = strtol(tokens[3], NULL, 10) - 1;
			Vector3 v0 = *((Vector3 *) arraylist_get(tmp->vertices, i0));
			Vector3 v1 = *((Vector3 *) arraylist_get(tmp->vertices, i1));
			Vector3 v2 = *((Vector3 *) arraylist_get(tmp->vertices, i2));
			AABB* aabb = (AABB *) calloc(1, sizeof(AABB));
			*aabb = aabb_create();
			aabb_update(aabb, v0);
			aabb_update(aabb, v1);
			aabb_update(aabb, v2);
			arraylist_add(tmp->aabbs, aabb);
			arraylist_add(tmp->triangles, triangle_new(v0, v1, v2));
			split_string_finish(tokens, FACE_COMPONENTS);
		}

		if (token) 
		{
			free(token);
		}
	}
	if (tokens)
	{
		free(tokens);
	}

	fclose(file);
}

static void copy_triangles(tmp_mesh* tmp, Mesh* mesh)
{
	mesh->triangles = (Triangle *) calloc(tmp->triangles->length, sizeof(Triangle));
	if (mesh->triangles)
	{
		for (int i = 0; i < tmp->triangles->length; i++)
		{
			Triangle* triangle = (Triangle *) arraylist_get(tmp->triangles, i);
			mesh->triangles[i] = *triangle;
			triangle_free(triangle);
		}
		mesh->triangle_amount = tmp->triangles->length;
		arraylist_free(tmp->triangles);
	}
	else
	{
		printf("mesh_load_obj: allocation of mesh tris failed");
	}

}

static void build_mesh_bounds(tmp_mesh* tmp, Mesh* mesh)
{
	mesh->aabb = aabb_create();
	for (int i = 0; i < tmp->vertices->length; i++)
	{
		Vector3* v = (Vector3 *) arraylist_get(tmp->vertices, i);
		aabb_update(&mesh->aabb, *v);
		vector3_free(v);
	}
	arraylist_free(tmp->vertices);
}

Mesh* mesh_new(const char* path)
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;

	tmp_mesh tmp;
	load_obj(mesh, path, &tmp);
	copy_triangles(&tmp, mesh);
	build_mesh_bounds(&tmp, mesh);

	for (int i = 0; i < tmp.aabbs->length; i++)
	{
		free((AABB *) arraylist_get(tmp.aabbs, i));
	}
	arraylist_free(tmp.aabbs);
	return mesh;
}

__device__
static int point_in_triangle(float bx, float by, float cx, float cy, float px, float py, float area)
{
	float ABP = tri_area_times_two(0, 0, bx, by, px, py);
	float BCP = tri_area_times_two(bx, by, cx, cy, px, py);
	float CAP = tri_area_times_two(cx, cy, 0, 0, px, py);
	return (ABP + BCP + CAP < area);
}

__device__
static float triangle_ray_intersect(Triangle tri, Ray ray)
{
	float d = vector3_dot(tri.n, vector3_sub(tri.v0, ray.o)) / vector3_dot(tri.n, ray.d);
	if (d < 0)
	{
		return -FLT_MAX;
	}
	Vector3 point = ray_position_along(ray, d);
	Vector3 p0 = vector3_sub(point, tri.v0);

	// one point of triangle is at origin 
	if (point_in_triangle(tri.t1x, tri.t1y, tri.t2x, tri.t2y,
						  vector3_dot(p0, tri.ex), vector3_dot(p0, tri.ey), tri.area))
	{
		return d;
	}

	return -FLT_MAX;
}

__device__
Hit mesh_ray_intersect(Mesh* mesh, Ray ray)
{
	Hit min = hit_create_no_intersect();
	min.d = FLT_MAX;
	for (int i = 0; i < mesh->triangle_amount; i++)
	{
		Triangle tri = mesh->triangles[i];
		float inter_d = triangle_ray_intersect(tri, ray);

		if (inter_d > 0 && inter_d < min.d)
		{
			min.d = inter_d;
			min.normal = tri.n;
			min.is_intersect = 1;
		}
	}

	return min;
}