 #include "mesh.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <map>

#include "../arraylist.h"
#include "../math/mathutils.h"
#include "../jsonutils.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const int NORMAL_COMPONENTS = 4;
static const int TEXCOORDS_COMPONENTS = 3;

struct OBJTri
{
    int pos[3];
    int tex[3];
    int norm[3];

    void add_vertex(int i, int p, int uv, int n)
    {
        pos[i] = p;  tex[i] = uv;  norm[i] = n;
    }
};

struct OBJMesh
{
    std::vector<Vector3> vertices;
    std::vector<Vector3> normals;
    std::vector<Vec2f> texcoords;

    std::vector<OBJTri> triangles;

    bool z_up, neg_z, face_normals, has_texcoords;
};

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

int count_occurences(char* string, char value)
{
    int count = 0;
    for (int i = 0; i < strlen(string); i++)
    {
        if (string[i] == value) count++;
    }
    return count;
}

OBJTri parse_face(char** tokens)
{
    OBJTri tri;
    char** vertices = (char**)calloc(3, sizeof(char *));
    for (int i = 1; i < 4; i++)
    {
        split_string(tokens[i], "/", 3, vertices);
        int slashes = count_occurences(tokens[i], '/');
        if (slashes == 0)
        {
            tri.add_vertex(i - 1, strtol(vertices[0], NULL, 10) - 1, -1, -1);
        }
        else if (slashes == 1)
        {
            tri.add_vertex(i - 1, strtol(vertices[0], NULL, 10) - 1, strtol(vertices[1], NULL, 10) - 1, -1);
        }
        else if (slashes == 2)
        {
            if (strstr(tokens[i], "//"))
            {
                tri.add_vertex(i - 1, strtol(vertices[0], NULL, 10) - 1, -1,
                    strtol(vertices[1], NULL, 10) - 1);
            }
            else
            {
                tri.add_vertex(i - 1, strtol(vertices[0], NULL, 10) - 1, strtol(vertices[1], NULL, 10) - 1,
                    strtol(vertices[2], NULL, 10) - 1);
            }
        }
        else
        {
            printf("parse_face: invalid face token %s\n", tokens[i]);
            tri.add_vertex(i - 1, -1, -1, -1);
        }
        split_string_finish(vertices, 3);
    }
    free(vertices);
    return tri;
}

static void load_obj(const char* path, OBJMesh* obj)
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

		if (strcmp(token, "v") == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			if (obj->z_up)
			{
                obj->vertices.push_back(vector3_create(strtof(tokens[1], NULL),
                    strtof(tokens[3], NULL),
                    strtof(tokens[2], NULL)));
			}
			else
			{
                obj->vertices.push_back(vector3_create(strtof(tokens[1], NULL),
                    strtof(tokens[2], NULL),
                    strtof(tokens[3], NULL)));
			}
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
        else if (strcmp(token, "vt") == 0)
        {
            split_string(line, " ", TEXCOORDS_COMPONENTS, tokens);
            obj->texcoords.push_back(Vec2f(strtof(tokens[1], NULL), strtof(tokens[2], NULL)));
            split_string_finish(tokens, TEXCOORDS_COMPONENTS);
        }
        else if (strcmp(token, "vn") == 0)
        {
            split_string(line, " ", NORMAL_COMPONENTS, tokens);
            obj->normals.push_back(vector3_create(strtof(tokens[1], NULL), 
                strtof(tokens[2], NULL), strtof(tokens[3], NULL)));
            split_string_finish(tokens, NORMAL_COMPONENTS);
        }
		else if (strcmp(token, "f") == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
            obj->triangles.push_back(parse_face(tokens));
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

struct PackedVertex
{
    Vector3 pos, norm;
    Vec2f uv;
};

struct VertexCompare
{
    bool operator()(const PackedVertex& a, const PackedVertex& b) const
    {
        if (a.pos.x < b.pos.x) return true;
        else if (a.pos.x > b.pos.x) return false;
        if (a.pos.y < b.pos.y) return true;
        else if (a.pos.y > b.pos.y) return false;
        if (a.pos.z < b.pos.z) return true;
        else if (a.pos.z > b.pos.z) return false;

        if (a.norm.x < b.norm.x) return true;
        else if (a.norm.x > b.norm.x) return false;
        if (a.norm.y < b.norm.y) return true;
        else if (a.norm.y > b.norm.y) return false;
        if (a.norm.z < b.norm.z) return true;
        else if (a.norm.z > b.norm.z) return false;

        if (a.uv.x < b.uv.x) return true;
        else if (a.uv.x > b.uv.x) return false;
        if (a.uv.y < b.uv.y) return true;
        else if (a.uv.y > b.uv.y) return false;

        return false;
    }
};

static void merge_vertices(OBJMesh* obj, Mesh* mesh)
{
    int merged_vertices = 0;
    std::vector<PackedVertex> vertex_buffer;
    std::map<PackedVertex, int, VertexCompare> vertex_map;
    mesh->triangles = new Triangle[obj->triangles.size()];
    for (int i = 0; i < obj->triangles.size(); i++)
    {
        Triangle tri;
        OBJTri& objtri = obj->triangles[i];
        for (int j = 0; j < 3; j++)
        {
            PackedVertex vert;
            vert.pos = obj->vertices[objtri.pos[j]];

            if (!obj->face_normals)
            {
                vert.norm = obj->normals[objtri.norm[j]];
                vector3_normalize(&vert.norm);
            }
            else
            {
                vert.norm = vector3_create(0, 0, 0);
            }

            if (obj->has_texcoords)
            {
                vert.uv = obj->texcoords[objtri.tex[j]];
            }
            else
            {
                vert.uv = Vec2f(0, 0);
            }

            int index;
            std::map<PackedVertex, int>::iterator it = vertex_map.find(vert);
            if (it == vertex_map.end())
            {
                index = vertex_buffer.size();
                vertex_map[vert] = index;
                vertex_buffer.push_back(vert);
            }
            else
            {
                index = it->second;
                merged_vertices++;
            }

            tri.indices[j] = index;
        }
        mesh->triangles[i] = tri;

        //printf("%d %d %d\n", tri.indices[0], tri.indices[1], tri.indices[2]);
    }

    mesh->positions = new Vector3[vertex_buffer.size()];
    mesh->normals = obj->face_normals ? nullptr : new Vector3[vertex_buffer.size()];
    mesh->texcoords = obj->has_texcoords ? new Vec2f[vertex_buffer.size()] : nullptr;

    for (int i = 0; i < vertex_buffer.size(); i++)
    {
        mesh->positions[i] = vertex_buffer[i].pos;
        if (obj->has_texcoords)
        {
            mesh->texcoords[i] = vertex_buffer[i].uv;
        }
        if (!obj->face_normals)
        {
            mesh->normals[i] = vertex_buffer[i].norm;
        }
        //printf("%f %f %f\n", mesh->positions[i].x, mesh->positions[i].y, mesh->positions[i].z);
    }
    mesh->vertex_amount = vertex_buffer.size();
    mesh->triangle_amount = obj->triangles.size();

    printf("collapse_obj: %d vertices merged\n", merged_vertices);
}

AABB get_bounds(Vector3* vertices, Triangle* tris, int tri_amount, 
                std::vector<AABB>& tri_bounds)
{
    AABB bounds;
    for (int i = 0; i < tri_amount; i++)
    {
        AABB tri_aabb = aabb_create();
        for (int j = 0; j < 3; j++)
        {
            aabb_update(&tri_aabb, vertices[tris[i].indices[j]]);
            aabb_update(&bounds, vertices[tris[i].indices[j]]);
        }
        //printf("%f %f %f == %f %f %f\n", tri_aabb.min.x, tri_aabb.min.y, tri_aabb.min.z, tri_aabb.max.x, tri_aabb.max.y, tri_aabb.max.z);
        //printf("%d %d %d\n", tris[i].indices[0], tris[i].indices[1], tris[i].indices[2]);
        fix_aabb(&tri_aabb);
        tri_bounds.push_back(tri_aabb);
    }
    fix_aabb(&bounds);
    return bounds;
}

void update_normals(Mesh* mesh, bool face_normals, bool flip_normals)
{
    if (face_normals)
    {
        if (flip_normals)
        {
            for (int i = 0; i < mesh->triangle_amount; i++)
            {
                Triangle& tri = mesh->triangles[i];
                int tmp = tri.indices[1];
                tri.indices[1] = tri.indices[2];
                tri.indices[2] = tmp;
            }
        }
    }
    else
    {
        if (flip_normals)
        {
            for (int i = 0; i < mesh->vertex_amount; i++)
            {
                mesh->normals[i] = vector3_mul(mesh->normals[i], -1.0f);
            }
        }
    }
}

Mesh mesh_create(const char* path, bool zUp, bool negZ, bool flipNormals, bool hasTexcoords, 
                 bool faceNormals, int tris_per_node)
{
	Mesh mesh;
	OBJMesh obj;
    obj.z_up = zUp;
    obj.neg_z = negZ;
    obj.face_normals = faceNormals;
    obj.has_texcoords = hasTexcoords;

	load_obj(path, &obj);
    merge_vertices(&obj, &mesh);

    update_normals(&mesh, faceNormals, flipNormals);

    std::vector<AABB> tri_aabbs;
    mesh.aabb = get_bounds(mesh.positions, mesh.triangles, mesh.triangle_amount, tri_aabbs);
	int length = strlen(path);
	char* filename = (char *)calloc(length + 1, sizeof(char));
	memcpy(filename, path, length);
	filename[length - 3] = 'b';
	filename[length - 2] = 'v';
	filename[length - 1] = 'h';
	filename[length] = '\0';
	mesh.bvh = bvh_create(tri_aabbs, mesh.aabb, filename, tris_per_node);

    mesh.has_texcoords = hasTexcoords;
    mesh.face_normals = faceNormals;
	return mesh;
}

void mesh_destroy(Mesh& mesh)
{
    bvh_free(&mesh.bvh);
    delete[] mesh.normals;
    delete[] mesh.positions;
    delete[] mesh.texcoords;
    delete[] mesh.triangles;
}

Mesh mesh_from_json(rapidjson::Value& json)
{
	std::string file;
	int itemsPerNode;
	bool zUp, negZ, flipNormals, hasTexcoords, faceNormals;
	JsonUtils::from_json(json, "file",               file);
	JsonUtils::from_json(json, "z_up",               zUp, false);
	JsonUtils::from_json(json, "neg_z",              negZ, false);
	JsonUtils::from_json(json, "flip_normals",       flipNormals, false);
    JsonUtils::from_json(json, "has_texcoords",      hasTexcoords, false);
    JsonUtils::from_json(json, "face_normals",       faceNormals, true);
	JsonUtils::from_json(json, "bvh_items_per_node", itemsPerNode, 4);
	return mesh_create(file.c_str(), zUp, negZ, flipNormals, hasTexcoords, faceNormals, itemsPerNode);
}

// Moller-trumbore method
__device__
static float triangle_ray_intersect(Vector3 v0, Vector3 v1, Vector3 v2, Ray ray, float& u, float& v)
{
	Vector3 P, Q, T, e1, e2;
	float det;

    e1 = vector3_sub(v1, v0);
    e2 = vector3_sub(v2, v0);

	P = vector3_cross(ray.d, e2);
	det = vector3_dot(e1, P);
	if (fabsf(det) < FLT_EPSILON)
	{
		return -1;
	}
	det = 1.f / det;
	T = vector3_sub(ray.o, v0);
	u = vector3_dot(T, P) * det;
	if (u < 0.f || u > 1.f)
	{
		return -1;
	}
	Q = vector3_cross(T, e1);
	v = vector3_dot(ray.d, Q) * det;
	if (v < 0.f || u + v > 1.f)
	{
		return -1;
	}

	return vector3_dot(e2, Q) * det;
}

// Based on method described at http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html
__device__ 
void mesh_ray_intersect(Mesh* mesh, Ray ray, Hit* hit)
{
	BVH* bvh = &mesh->bvh;
	Triangle* tris = mesh->triangles;

	int stack[BVH_STACK_SIZE];
	int stack_idx = 0;
	stack[stack_idx++] = 0;
	hit_set_no_intersect(hit);
	hit->d = FLT_MAX;
    int tri_index = -1;

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
				float d = triangle_ray_intersect(mesh->positions[tri->indices[0]], 
                                                 mesh->positions[tri->indices[1]], 
                                                 mesh->positions[tri->indices[2]], 
                                                 ray, u, v);
				if (d > 0 && d < hit->d)
				{
                    tri_index = bvh->tri_indices[i];
                    hit->uv = Vec2f(u, v);
					hit->d = d;
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

    if (tri_index != -1)
    {
        Triangle* tri = &tris[tri_index];

        if (mesh->face_normals)
        {
            Vector3 v0 = mesh->positions[tri->indices[0]],
                v1 = mesh->positions[tri->indices[1]],
                v2 = mesh->positions[tri->indices[2]];
            hit->normal = vector3_cross(vector3_sub(v1, v0), vector3_sub(v2, v0));
            vector3_normalize(&hit->normal);
        }
        else
        {
            Vector3 n0 = mesh->normals[tri->indices[0]],
                n1 = mesh->normals[tri->indices[1]],
                n2 = mesh->normals[tri->indices[2]];
            hit->normal = vector3_add(vector3_mul(n0, 1.0f - hit->uv.x - hit->uv.y),
                vector3_add(vector3_mul(n1, hit->uv.x), vector3_mul(n2, hit->uv.y)));
            vector3_normalize(&hit->normal);
        }

        if (mesh->has_texcoords)
        {
            Vec2f uv0 = mesh->texcoords[tri->indices[0]];
            Vec2f uv1 = mesh->texcoords[tri->indices[1]];
            Vec2f uv2 = mesh->texcoords[tri->indices[2]];
            hit->uv = uv0 * (1.0f - hit->uv.x - hit->uv.y) + uv1 * hit->uv.x + uv2 * hit->uv.y;
        }
    }
}