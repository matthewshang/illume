#pragma once

#include "scene.h"
#include "../primitives/sphere.h"
#include "../primitives/mesh.h"
#include "../primitives/mesh_instance.h"
#include "../camera.h"
#include "../math/vector3.h"
#include "../math/vector2.h"

class SceneRef
{
public:
    SceneRef(HostScene& scene);
    ~SceneRef();

    inline DeviceScene* getScene() { return m_scene; };

private:
    // All pointers in device space
    DeviceScene* m_scene;
    
    Sphere* m_spheres;
    Mesh* m_meshes;
    int m_mesh_amount;
    MeshInstance* m_instances;

    Triangle** m_triangles;
    bool* m_has_texcoords;
    Vec2f** m_texcoords;
    GPUNode** m_bvh_nodes;
    int** m_bvh_indices;
    Vector3** m_vertices;
    bool* m_face_normals;
    Vector3** m_normals;
};