#ifndef _MESH_
#define _MESH_

#include <stdio.h>
#include <string.h>

#include "triangle.h"
#include "../arraylist.h"
#include "../math/vector3.h"

static const int OBJ_TOKENS = 4;
static const int VERTEX_COMPONENTS = 4;
static const int FACE_COMPONENTS = 4;
static const char* TOKEN_VERTEX = "v";
static const char* TOKEN_FACE = "f";

typedef struct
{
	int triangle_amount;
	Triangle* triangles;
}
Mesh;

#ifdef __cplusplus
extern "C" {
#endif


Mesh  mesh_new       ();
void  mesh_load_obj  (Mesh* mesh, const char* path);
void  mesh_free      (Mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif