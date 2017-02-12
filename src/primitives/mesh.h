#ifndef _MESH_
#define _MESH_

#include "rapidjson/document.h"

#include "../math/ray.h"
#include "../math/aabb.h"
#include "../math/vector3.h"
#include "../math/vector2.h"
#include "../hit.h"
#include "../accel/bvh.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const int TEXCOORDS_COMPONENTS = 3;
static const char* TOKEN_VERTEX = "v";
static const char* TOKEN_FACE = "f";
static const char* TOKEN_TEXCOORDS = "vt";

typedef struct
{
	Vector3 e1;
	Vector3 e2;
	Vector3 v0;
	Vector3 n;
    int indices[3];
}
Triangle;

typedef struct
{
    bool has_texcoords;
	int triangle_amount;
	Triangle* triangles;
    int vertex_amount;
    Vec2f* texcoords;
	AABB aabb;
	BVH bvh;
}
Mesh;

Mesh   mesh_create         (const char* path, bool zUp, bool negZ, bool flipNormals, bool hasTexcoords, int tris_per_node);
Mesh   mesh_from_json      (rapidjson::Value& json);
void   mesh_free           (Mesh* mesh);
__device__
void   mesh_ray_intersect  (Mesh* mesh, Ray ray, Hit* hit);

#endif