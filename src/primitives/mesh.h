#ifndef _MESH_
#define _MESH_

#include "rapidjson/document.h"

#include "../math/ray.h"
#include "../math/aabb.h"
#include "../math/vector3.h"
#include "../math/vector2.h"
#include "../hit.h"
#include "../accel/bvh.h"

struct Triangle
{
    int indices[3];
};

struct Mesh
{
    AABB aabb;
    BVH bvh;

	int triangle_amount;
	Triangle* triangles;

    bool has_texcoords;
    bool face_normals;
    int vertex_amount;
    Vector3* positions;
    Vector3* normals;
    Vec2f* texcoords;
    //bool a;
};

Mesh   mesh_create         (const char* path, bool zUp, bool negZ, bool flipNormals, 
                            bool hasTexcoords, bool faceNormals, int tris_per_node);
Mesh   mesh_from_json      (rapidjson::Value& json);
void   mesh_destroy        (Mesh& mesh);
void   mesh_free           (Mesh* mesh);
__device__
void   mesh_ray_intersect  (Mesh* mesh, Ray ray, Hit* hit);

#endif