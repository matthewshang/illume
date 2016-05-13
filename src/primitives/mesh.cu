#include "mesh.h"

static void load_obj(Mesh* mesh, const char* path);

Mesh* mesh_new(const char* path)
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;
	load_obj(mesh, path);
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
		kdtree_free(&mesh->tree);
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

static void load_obj(Mesh* mesh, const char* path)
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
	ArrayList* aabbs = arraylist_new(1);

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			arraylist_add(vertices, vector3_new(strtof(tokens[1], NULL),
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
			Vector3 v0 = *((Vector3 *) arraylist_get(vertices, i0));
			Vector3 v1 = *((Vector3 *) arraylist_get(vertices, i1));
			Vector3 v2 = *((Vector3 *) arraylist_get(vertices, i2));
			AABB* aabb = (AABB *) calloc(1, sizeof(AABB));
			*aabb = aabb_create();
			aabb_update(aabb, v0);
			aabb_update(aabb, v1);
			aabb_update(aabb, v2);
			arraylist_add(aabbs, aabb);
			arraylist_add(triangles, triangle_new(v0, v1, v2));
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

	mesh->aabb = aabb_create();
	for (int i = 0; i < vertices->length; i++)
	{
		Vector3* v = (Vector3 *) arraylist_get(vertices, i);
		aabb_update(&mesh->aabb, *v); 
		vector3_free(v);
	}
	arraylist_free(vertices);
	AABB* final_aabbs = (AABB *) calloc(aabbs->length, sizeof(AABB));
	for (int i = 0; i < triangles->length; i++)
	{
		triangle_free((Triangle *) arraylist_get(triangles, i));
		AABB* current = (AABB *) arraylist_get(aabbs, i);
		final_aabbs[i] = *current;
		free(current);
	}
	arraylist_free(triangles);
	mesh->tree = kdtree_build(final_aabbs, aabbs->length, mesh->aabb, 5, 5);
	arraylist_free(aabbs);
	free(final_aabbs);
	fclose(file);
	if (tokens)
	{ 
		free(tokens);
	}
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
		return -1;
	}
	Vector3 point = ray_position_along(ray, d);
	Vector3 p0 = vector3_sub(point, tri.v0);

	// one point of triangle is at origin 
	if (point_in_triangle(tri.t1x, tri.t1y, tri.t2x, tri.t2y,
						  vector3_dot(p0, tri.ex), vector3_dot(p0, tri.ey), tri.area))
	{
		return d;
	}

	return -1;
}

__device__
Hit mesh_ray_intersect(Mesh* mesh, Ray ray)
{
	Hit min = hit_create_no_intersect();
	min.d = FLOAT_MAX;
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