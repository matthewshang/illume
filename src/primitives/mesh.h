#ifndef _MESH_
#define _MESH_

#include <stdio.h>
#include <string.h>
#include <float.h>

#include "triangle.h"
#include "../material.h"
#include "../arraylist.h"
#include "../math/vector3.h"
#include "../math/ray.h"
#include "../intersection.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const char* TOKEN_VERTEX = "v";
static const char* TOKEN_FACE = "f";

typedef struct
{
	Material m;
	int triangle_amount;
	Triangle* triangles;
}
Mesh;

#ifdef __cplusplus
extern "C" {
#endif


            Mesh*         mesh_new            (Material m);
            void          mesh_free           (Mesh* mesh);
            void          mesh_load_obj       (Mesh* mesh, const char* path);
__device__  Intersection  mesh_ray_intersect  (Mesh mesh, Ray ray);

#ifdef __cplusplus
}
#endif

#endif