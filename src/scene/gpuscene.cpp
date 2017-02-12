#include "gpuscene.h"

#include "../intellisense.h"
#include "../error_check.h"

// Copies the host scene to device memory by modifying the already allocated structure 
// to hold device pointers, copying, and then putting the host pointers back

GPUScene::GPUScene(Scene* scene)
{
    // mesh triangles
    m_triangles = new Triangle*[scene->mesh_amount];
    Triangle** tmp_triangles = new Triangle*[scene->mesh_amount];
    m_bvh_nodes = new GPUNode*[scene->mesh_amount];
    GPUNode** tmp_nodes = new GPUNode*[scene->mesh_amount];
    m_bvh_indices = new int*[scene->mesh_amount];
    int** tmp_indices = new int*[scene->mesh_amount];
    m_texcoords = new Vec2f*[scene->mesh_amount];
    m_has_texcoords = new bool[scene->mesh_amount];
    Vec2f** tmp_texcoords = new Vec2f*[scene->mesh_amount];

    for (int i = 0; i < scene->mesh_amount; i++)
    {
        Mesh* mesh = &scene->meshes[i];
        int triangles_size = mesh->triangle_amount * sizeof(Triangle);
        HANDLE_ERROR( cudaMalloc(&m_triangles[i], triangles_size) );
        HANDLE_ERROR( cudaMemcpy(m_triangles[i], mesh->triangles, triangles_size, cudaMemcpyHostToDevice) );
        tmp_triangles[i] = mesh->triangles;
        mesh->triangles = m_triangles[i];

        int nodes_size = mesh->bvh.node_amount * sizeof(GPUNode);
        HANDLE_ERROR( cudaMalloc(&m_bvh_nodes[i], nodes_size) );
        HANDLE_ERROR( cudaMemcpy(m_bvh_nodes[i], mesh->bvh.nodes, nodes_size, cudaMemcpyHostToDevice) );
        tmp_nodes[i] = mesh->bvh.nodes;
        mesh->bvh.nodes = m_bvh_nodes[i];

        int indices_size = mesh->bvh.tri_index_amount * sizeof(int);
        HANDLE_ERROR( cudaMalloc(&m_bvh_indices[i], indices_size) );
        HANDLE_ERROR( cudaMemcpy(m_bvh_indices[i], mesh->bvh.tri_indices, indices_size, cudaMemcpyHostToDevice) );
        tmp_indices[i] = mesh->bvh.tri_indices;
        mesh->bvh.tri_indices = m_bvh_indices[i];

        m_has_texcoords[i] = mesh->has_texcoords;
        if (mesh->has_texcoords)
        {
            int texcoords_size = mesh->vertex_amount * sizeof(Vec2f);
            HANDLE_ERROR( cudaMalloc(&m_texcoords[i], texcoords_size) );
            HANDLE_ERROR( cudaMemcpy(m_texcoords[i], mesh->texcoords, texcoords_size, cudaMemcpyHostToDevice) );
            tmp_texcoords[i] = mesh->texcoords;
            mesh->texcoords = m_texcoords[i];
        }
    }

    // scene
    int spheres_size = sizeof(Sphere) * scene->sphere_amount;
    m_mesh_amount = scene->mesh_amount;
    int meshes_size = sizeof(Mesh) * scene->mesh_amount;
    int instances_size = sizeof(MeshInstance) * scene->instance_amount;

    HANDLE_ERROR( cudaMalloc(&m_spheres, spheres_size) );
    HANDLE_ERROR( cudaMalloc(&m_meshes, meshes_size) );
    HANDLE_ERROR( cudaMalloc(&m_instances, instances_size) );

    Sphere* tmp_spheres = scene->spheres;
    Mesh* tmp_meshes = scene->meshes;
    MeshInstance* tmp_instances = scene->instances;
    scene->spheres = m_spheres;
    scene->meshes = m_meshes;
    scene->instances = m_instances;
    HANDLE_ERROR( cudaMalloc(&m_scene, sizeof(Scene)) );
    HANDLE_ERROR( cudaMemcpy(m_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice) );
    scene->spheres = tmp_spheres;
    scene->meshes = tmp_meshes;
    scene->instances = tmp_instances;
    HANDLE_ERROR( cudaMemcpy(m_spheres, scene->spheres, spheres_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_meshes, scene->meshes, meshes_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_instances, scene->instances, instances_size, cudaMemcpyHostToDevice) );

    // cleanup
    for (int i = 0; i < scene->mesh_amount; i++)
    {
        scene->meshes[i].triangles = tmp_triangles[i];
        scene->meshes[i].bvh.nodes = tmp_nodes[i];
        scene->meshes[i].bvh.tri_indices = tmp_indices[i];
        if (scene->meshes[i].has_texcoords) scene->meshes[i].texcoords = tmp_texcoords[i];
    }
    delete tmp_triangles;
    delete tmp_nodes;
    delete tmp_indices;
    delete tmp_texcoords;
}

GPUScene::~GPUScene()
{
    HANDLE_ERROR( cudaFree(m_spheres) );
    HANDLE_ERROR( cudaFree(m_meshes) );
    HANDLE_ERROR( cudaFree(m_instances) );
    HANDLE_ERROR( cudaFree(m_scene) );
    for (int i = 0; i < m_mesh_amount; i++)
    {
        HANDLE_ERROR( cudaFree(m_triangles[i]) );
        HANDLE_ERROR( cudaFree(m_bvh_nodes[i]) );
        HANDLE_ERROR( cudaFree(m_bvh_indices[i]) );
        if (m_has_texcoords[i]) HANDLE_ERROR( cudaFree(m_texcoords[i]) );
    }
    delete m_triangles;
    delete m_bvh_nodes;
    delete m_bvh_indices;
    delete m_texcoords;
    delete m_has_texcoords;
}
