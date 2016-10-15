#ifndef _MESH_
#define _MESH_

#include "../math/ray.h"
#include "../math/aabb.h"
#include "../math/vector3.h"
#include "../hit.h"
#include "../accel/bvh.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const char* TOKEN_VERTEX = "v";
static const char* TOKEN_FACE = "f";

typedef struct
{
	Vector3 e1;
	Vector3 e2;
	Vector3 v0;
	Vector3 n;
}
Triangle;

typedef struct
{
	int triangle_amount;
	Triangle* triangles;
	AABB aabb;
	BVH bvh;
}
Mesh;

#ifdef __cplusplus
extern "C" {
#endif


            Mesh*  mesh_new            (const char* path, int zUp, int tris_per_node);
            void   mesh_free           (Mesh* mesh);
__device__  Hit    mesh_ray_intersect  (Mesh* mesh, Ray ray);

#ifdef __cplusplus
}
#endif

#endif