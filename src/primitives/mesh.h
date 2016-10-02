#ifndef _MESH_
#define _MESH_

#include "../math/ray.h"
#include "../math/aabb.h"
#include "../hit.h"
#include "triangle.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const char* TOKEN_VERTEX = "v";
static const char* TOKEN_FACE = "f";

typedef struct
{
	int triangle_amount;
	Triangle* triangles;
	AABB aabb;
}
Mesh;

#ifdef __cplusplus
extern "C" {
#endif


            Mesh*  mesh_new            (const char* path);
            void   mesh_free           (Mesh* mesh);
__device__  Hit    mesh_ray_intersect  (Mesh* mesh, Ray ray);

#ifdef __cplusplus
}
#endif

#endif