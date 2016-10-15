 #include "mesh.h"

#include <stdio.h>
#include <string.h>

#include "../arraylist.h"
#include "../math/constants.h"

#define BVH_STACK_SIZE 32

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
	tri.e1 = vector3_sub(v1, v0);
	tri.e2 =  vector3_sub(v2, v0);
	tri.v0 = v0;
	tri.n = vector3_cross(tri.e1, tri.e2);
	vector3_normalize(&tri.n);
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

static void load_obj(Mesh* mesh, const char* path, tmp_mesh* tmp, int zUp)
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
	int tris = 0;

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			if (zUp)
			{
				arraylist_add(tmp->vertices, vector3_new(strtof(tokens[1], NULL),
					strtof(tokens[3], NULL),
					strtof(tokens[2], NULL)));
			}
			else
			{
				arraylist_add(tmp->vertices, vector3_new(strtof(tokens[1], NULL),
					strtof(tokens[2], NULL),
					strtof(tokens[3], NULL)));
			}
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
			fix_aabb(aabb);
			arraylist_add(tmp->aabbs, aabb);
			arraylist_add(tmp->triangles, triangle_new(v0, v1, v2));
			split_string_finish(tokens, FACE_COMPONENTS);
			tris++;
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
	printf("Loaded mesh from %s...\n", path);
	printf("Triangles: %d\n", tris);
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
		}
		mesh->triangle_amount = tmp->triangles->length;
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
	}
}

Mesh* mesh_new(const char* path, int zUp)
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;

	tmp_mesh tmp;
	tmp.vertices = arraylist_new(3);
	tmp.triangles = arraylist_new(1);
	tmp.aabbs = arraylist_new(1);
	load_obj(mesh, path, &tmp, zUp);
	copy_triangles(&tmp, mesh);
	build_mesh_bounds(&tmp, mesh);
	int length = strlen(path);
	char* filename = (char *) calloc(length + 1, sizeof(char));
	memcpy(filename, path, length);
	filename[length - 3] = 'b';
	filename[length - 2] = 'v';
	filename[length - 1] = 'h';
	filename[length] = '\0';
	mesh->bvh = bvh_create(tmp.aabbs, mesh->aabb, filename);

	for (int i = 0; i < tmp.aabbs->length; i++)
	{
		free((AABB *) arraylist_get(tmp.aabbs, i));
	}
	arraylist_free(tmp.aabbs);
	for (int i = 0; i < tmp.vertices->length; i++)
	{
		free(arraylist_get(tmp.vertices, i));
	}
	arraylist_free(tmp.vertices);
	for (int i = 0; i < tmp.triangles->length; i++)
	{
		free(arraylist_get(tmp.triangles, i));
	}
	arraylist_free(tmp.triangles);
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

// Moller-trumbore method
__device__
static float triangle_ray_intersect(Triangle tri, Ray ray)
{
	Vector3 P, Q, T;
	float det, u, v;

	P = vector3_cross(ray.d, tri.e2);
	det = vector3_dot(tri.e1, P);
	if (fabsf(det) < FLT_EPSILON)
	{
		return -1;
	}
	det = 1.f / det;
	T = vector3_sub(ray.o, tri.v0);
	u = vector3_dot(T, P) * det;
	if (u < 0.f || u > 1.f)
	{
		return -1;
	}
	Q = vector3_cross(T, tri.e1);
	v = vector3_dot(ray.d, Q) * det;
	if (v < 0.f || u + v > 1.f)
	{
		return -1;
	}
	return vector3_dot(tri.e2, Q) * det;
}

// Based on method described at http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html
__device__
Hit bvh_ray_intersect(Triangle* tris, BVH* bvh, Ray ray)
{
	int stack[BVH_STACK_SIZE];
	int stack_idx = 0;
	stack[stack_idx++] = 0;
	Hit min = hit_create_no_intersect();
	min.d = FLT_MAX;

	while (stack_idx > 0)
	{
		GPUNode node = bvh->nodes[stack[stack_idx - 1]];
		stack_idx--;
		if (node.u.leaf.tri_amount & 0x80000000)
		{
			for (unsigned i = node.u.leaf.tri_start;
				i < node.u.leaf.tri_start + (node.u.leaf.tri_amount & 0x7fffffff);
				i++)
				{
					Triangle tri = tris[bvh->tri_indices[i]];
					float d = triangle_ray_intersect(tri, ray);
					if (d > 0 && d < min.d)
					{
						min.d = d;
						min.normal = tri.n;
						min.is_intersect = 1;
					}
				}
		}
		else
		{
			if (aabb_ray_intersect(node.bounds, ray) != -FLT_MAX)
			{
				stack[stack_idx++] = node.u.node.right_node;
				stack[stack_idx++] = node.u.node.left_node;
				if (stack_idx > BVH_STACK_SIZE)
				{
					return hit_create_no_intersect();
				}
			}
		}
	}
	return min;
}

__device__
Hit mesh_ray_intersect(Mesh* mesh, Ray ray)
{
	//Hit min = hit_create_no_intersect();
	//min.d = FLT_MAX;
	//for (int i = 0; i < mesh->triangle_amount; i++)
	//{
	//	Triangle tri = mesh->triangles[i];
	//	float inter_d = triangle_ray_intersect(tri, ray);

	//	if (inter_d > 0 && inter_d < min.d)
	//	{
	//		min.d = inter_d;
	//		min.normal = tri.n;
	//		min.is_intersect = 1;
	//	}
	//}
	//
	//return min;
	return bvh_ray_intersect(mesh->triangles, &mesh->bvh, ray);
}