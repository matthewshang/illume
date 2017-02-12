#pragma once

#include "scene.h"

class GPUScene
{
public:
    GPUScene(Scene* scene);
    ~GPUScene();

    inline Scene* getDeviceScene() { return m_scene; };

private:
    // All pointers in device space
    Scene* m_scene;
    
    Sphere* m_spheres;
    Mesh* m_meshes;
    int m_mesh_amount;
    MeshInstance* m_instances;

    Triangle** m_triangles;
    bool* m_has_texcoords;
    Vec2f** m_texcoords;
    GPUNode** m_bvh_nodes;
    int** m_bvh_indices;
};