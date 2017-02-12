 #include "mesh.h"

#include <stdio.h>
#include <string.h>
#include <vector>

#include "../arraylist.h"
#include "../math/mathutils.h"
#include "../jsonutils.h"

typedef struct
{
    std::vector<Vector3> vertices;
    std::vector<Triangle> triangles;
    std::vector<AABB> aabbs;
    std::vector<Vec2f> texcoords;
}
tmp_mesh;

static Triangle triangle_create(Vector3 v0, Vector3 v1, Vector3 v2, 
                                int i0, int i1, int i2, bool flip_normals)
{
	Triangle tri;
	tri.e1 = vector3_sub(v1, v0);
	tri.e2 =  vector3_sub(v2, v0);
	tri.v0 = v0;
    tri.n = flip_normals ? vector3_cross(tri.e2, tri.e1) :
        vector3_cross(tri.e1, tri.e2);
	vector3_normalize(&tri.n);
    tri.indices[0] = i0;
    tri.indices[1] = i1;
    tri.indices[2] = i2;
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

static void split_string(const char* string, const char* delim, int tokens, char** ret, int offset = 0)
{
	char* dup = strdup(string);
	char* p = strtok(dup, delim);
	int i = 0;
	while (i < tokens && p != NULL)
	{
		ret[offset + i++] = strdup(p);
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

static void load_obj(Mesh* mesh, const char* path, tmp_mesh* tmp, bool zUp, 
    bool negZ, bool flipNormals, bool hasTexcoords)
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
                tmp->vertices.push_back(vector3_create(strtof(tokens[1], NULL),
                    strtof(tokens[3], NULL),
                    strtof(tokens[2], NULL)));
			}
			else
			{
                tmp->vertices.push_back(vector3_create(strtof(tokens[1], NULL),
                    strtof(tokens[2], NULL),
                    strtof(tokens[3], NULL)));
			}
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
        else if (strcmp(token, TOKEN_TEXCOORDS) == 0)
        {
            split_string(line, " ", TEXCOORDS_COMPONENTS, tokens);
            tmp->texcoords.push_back(Vec2f(strtof(tokens[1], NULL), strtof(tokens[2], NULL)));
            split_string_finish(tokens, TEXCOORDS_COMPONENTS);
        }
		else if (strcmp(token, TOKEN_FACE) == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
            int i0, i1, i2;
            if (hasTexcoords)
            {
                char** texindices = (char**)calloc(6, sizeof(char *));
                for (int i = 1; i < 4; i++)
                {
                    split_string(tokens[i], "/", 2, texindices, i * 2 - 2);
                }
                i0 = strtol(texindices[1], NULL, 10) - 1;
                i1 = strtol(texindices[3], NULL, 10) - 1;
                i2 = strtol(texindices[5], NULL, 10) - 1;
                split_string_finish(texindices, 6);
                free(texindices);
            }
            else
            {
                i0 = i1 = i2 = -1;
            }
			Vector3 v0 = tmp->vertices[strtol(tokens[1], NULL, 10) - 1];
			Vector3 v1 = tmp->vertices[strtol(tokens[2], NULL, 10) - 1];
			Vector3 v2 = tmp->vertices[strtol(tokens[3], NULL, 10) - 1];
			AABB aabb = aabb_create();
			aabb_update(&aabb, v0);
			aabb_update(&aabb, v1);
			aabb_update(&aabb, v2);
			fix_aabb(&aabb);
            tmp->aabbs.push_back(aabb);
            tmp->triangles.push_back(triangle_create(v0, v1, v2, i0, i1, i2, flipNormals));
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

static void copy_data(tmp_mesh* tmp, Mesh* mesh, bool hasTexcoords)
{
	mesh->triangles = (Triangle *) calloc(tmp->triangles.size(), sizeof(Triangle));
	if (mesh->triangles)
	{
        std::copy(tmp->triangles.begin(), tmp->triangles.end(), mesh->triangles);
		mesh->triangle_amount = tmp->triangles.size();
	}
	else
	{
		printf("copy_triangles: allocation of mesh tris failed\n");
	}

    if (hasTexcoords)
    {
        mesh->texcoords = (Vec2f *)calloc(tmp->texcoords.size(), sizeof(Vec2f));
        if (mesh->texcoords)
        {
            std::copy(tmp->texcoords.begin(), tmp->texcoords.end(), mesh->texcoords);
            mesh->vertex_amount = tmp->texcoords.size();
        }
        else
        {
            printf("copy_texcoords: allocation of texcoords failed\n");
        }
    }
}

static void build_mesh_bounds(tmp_mesh* tmp, Mesh* mesh)
{
	mesh->aabb = aabb_create();
	for (int i = 0; i < tmp->vertices.size(); i++)
	{
		aabb_update(&mesh->aabb, tmp->vertices[i]);
	}
}

Mesh mesh_create(const char* path, bool zUp, bool negZ, bool flipNormals, bool hasTexcoords, int tris_per_node)
{
	Mesh mesh;
	mesh.triangle_amount = 0;
	mesh.triangles = NULL;

	tmp_mesh tmp;
	load_obj(&mesh, path, &tmp, zUp, negZ, flipNormals, hasTexcoords);
	copy_data(&tmp, &mesh, hasTexcoords);
	build_mesh_bounds(&tmp, &mesh);
	//printf("mesh bounds: \nmin: %f %f %f\nmax: %f %f %f\n", mesh->aabb.min.x, mesh->aabb.min.y, mesh->aabb.min.z, mesh->aabb.max.x, mesh->aabb.max.y, mesh->aabb.max.z);
	int length = strlen(path);
	char* filename = (char *)calloc(length + 1, sizeof(char));
	memcpy(filename, path, length);
	filename[length - 3] = 'b';
	filename[length - 2] = 'v';
	filename[length - 1] = 'h';
	filename[length] = '\0';
	mesh.bvh = bvh_create(tmp.aabbs, mesh.aabb, filename, tris_per_node);
    mesh.has_texcoords = hasTexcoords;
	return mesh;
}

Mesh mesh_from_json(rapidjson::Value& json)
{
	std::string file;
	int itemsPerNode;
	bool zUp, negZ, flipNormals, hasTexcoords;
	JsonUtils::from_json(json, "file",               file);
	JsonUtils::from_json(json, "z_up",               zUp, false);
	JsonUtils::from_json(json, "neg_z",              negZ, false);
	JsonUtils::from_json(json, "flip_normals",       flipNormals, false);
    JsonUtils::from_json(json, "has_texcoords",      hasTexcoords, false);
	JsonUtils::from_json(json, "bvh_items_per_node", itemsPerNode, 4);
	return mesh_create(file.c_str(), zUp, negZ, flipNormals, hasTexcoords, itemsPerNode);
}

// Moller-trumbore method
__device__
static float triangle_ray_intersect(Triangle* tri, Ray ray, float& u_, float& v_)
{
	Vector3 P, Q, T;
	float det, u, v;

	P = vector3_cross(ray.d, tri->e2);
	det = vector3_dot(tri->e1, P);
	if (fabsf(det) < FLT_EPSILON)
	{
		return -1;
	}
	det = 1.f / det;
	T = vector3_sub(ray.o, tri->v0);
	u = vector3_dot(T, P) * det;
	if (u < 0.f || u > 1.f)
	{
		return -1;
	}
	Q = vector3_cross(T, tri->e1);
	v = vector3_dot(ray.d, Q) * det;
	if (v < 0.f || u + v > 1.f)
	{
		return -1;
	}
    u_ = u;
    v_ = v;

	return vector3_dot(tri->e2, Q) * det;
}

// Based on method described at http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

__device__ 
void mesh_ray_intersect(Mesh* mesh, Ray ray, Hit* hit)
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
	BVH* bvh = &mesh->bvh;
	Triangle* tris = mesh->triangles;

	int stack[BVH_STACK_SIZE];
	int stack_idx = 0;
	stack[stack_idx++] = 0;
	hit_set_no_intersect(hit);
	hit->d = FLT_MAX;

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
				Triangle* tri = &tris[bvh->tri_indices[i]];
                float u, v = 0;
				float d = triangle_ray_intersect(tri, ray, u, v);
				if (d > 0 && d < hit->d)
				{
                    if (mesh->has_texcoords)
                    {
                        Vec2f uv0 = mesh->texcoords[tri->indices[0]];
                        Vec2f uv1 = mesh->texcoords[tri->indices[1]];
                        Vec2f uv2 = mesh->texcoords[tri->indices[2]];
                        hit->uv = uv0 * (1.0f - u - v) + uv1 * u + uv2 * v;
                    }
					hit->d = d;
					hit->normal = tri->n;
					hit->is_intersect = true;
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
					hit_set_no_intersect(hit);
					return;
				}
			}
		}
	}
}