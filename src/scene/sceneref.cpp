#include "sceneref.h"

#include "../intellisense.h"
#include "../error_check.h"

// Copies the host scene to device memory by modifying the already allocated structure 
// to hold device pointers, copying, and then putting the host pointers back

SceneRef::SceneRef(HostScene& scene)
{
    // mesh triangles
    m_triangles = new Triangle*[scene.m_meshes.size()];
    Triangle** tmp_triangles = new Triangle*[scene.m_meshes.size()];
    m_bvh_nodes = new GPUNode*[scene.m_meshes.size()];
    GPUNode** tmp_nodes = new GPUNode*[scene.m_meshes.size()];
    m_bvh_indices = new int*[scene.m_meshes.size()];
    int** tmp_indices = new int*[scene.m_meshes.size()];
    m_texcoords = new Vec2f*[scene.m_meshes.size()];
    m_has_texcoords = new bool[scene.m_meshes.size()];
    Vec2f** tmp_texcoords = new Vec2f*[scene.m_meshes.size()];

    for (int i = 0; i < scene.m_meshes.size(); i++)
    {
        Mesh& mesh = scene.m_meshes[i];
        int triangles_size = mesh.triangle_amount * sizeof(Triangle);
        HANDLE_ERROR( cudaMalloc(&m_triangles[i], triangles_size) );
        HANDLE_ERROR( cudaMemcpy(m_triangles[i], mesh.triangles, triangles_size, cudaMemcpyHostToDevice) );
        tmp_triangles[i] = mesh.triangles;
        mesh.triangles = m_triangles[i];

        int nodes_size = mesh.bvh.node_amount * sizeof(GPUNode);
        HANDLE_ERROR( cudaMalloc(&m_bvh_nodes[i], nodes_size) );
        HANDLE_ERROR( cudaMemcpy(m_bvh_nodes[i], mesh.bvh.nodes, nodes_size, cudaMemcpyHostToDevice) );
        tmp_nodes[i] = mesh.bvh.nodes;
        mesh.bvh.nodes = m_bvh_nodes[i];

        int indices_size = mesh.bvh.tri_index_amount * sizeof(int);
        HANDLE_ERROR( cudaMalloc(&m_bvh_indices[i], indices_size) );
        HANDLE_ERROR( cudaMemcpy(m_bvh_indices[i], mesh.bvh.tri_indices, indices_size, cudaMemcpyHostToDevice) );
        tmp_indices[i] = mesh.bvh.tri_indices;
        mesh.bvh.tri_indices = m_bvh_indices[i];

        m_has_texcoords[i] = mesh.has_texcoords;
        if (mesh.has_texcoords)
        {
            int texcoords_size = mesh.vertex_amount * sizeof(Vec2f);
            HANDLE_ERROR( cudaMalloc(&m_texcoords[i], texcoords_size) );
            HANDLE_ERROR( cudaMemcpy(m_texcoords[i], mesh.texcoords, texcoords_size, cudaMemcpyHostToDevice) );
            tmp_texcoords[i] = mesh.texcoords;
            mesh.texcoords = m_texcoords[i];
        }
    }

    // scene
    DeviceScene tmp_scene;
    tmp_scene.camera = scene.m_camera;
    tmp_scene.envmap = scene.m_environment;
    tmp_scene.sphere_amount = scene.m_spheres.size();
    tmp_scene.mesh_amount = scene.m_meshes.size();
    tmp_scene.instance_amount = scene.m_instances.size();

    int spheres_size = sizeof(Sphere) * scene.m_spheres.size();
    m_mesh_amount = scene.m_meshes.size();
    int meshes_size = sizeof(Mesh) * m_mesh_amount;
    int instances_size = sizeof(MeshInstance) * scene.m_instances.size();

    HANDLE_ERROR( cudaMalloc(&m_spheres, spheres_size) );
    HANDLE_ERROR( cudaMalloc(&m_meshes, meshes_size) );
    HANDLE_ERROR( cudaMalloc(&m_instances, instances_size) );

    tmp_scene.spheres = m_spheres;
    tmp_scene.meshes = m_meshes;
    tmp_scene.instances = m_instances;
    HANDLE_ERROR( cudaMalloc(&m_scene, sizeof(DeviceScene)) );
    HANDLE_ERROR( cudaMemcpy(m_scene, &tmp_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_spheres, scene.m_spheres.data(), spheres_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_meshes, scene.m_meshes.data(), meshes_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_instances, scene.m_instances.data(), instances_size, cudaMemcpyHostToDevice) );

    // cleanup
    for (int i = 0; i < m_mesh_amount; i++)
    {
        scene.m_meshes[i].triangles = tmp_triangles[i];
        scene.m_meshes[i].bvh.nodes = tmp_nodes[i];
        scene.m_meshes[i].bvh.tri_indices = tmp_indices[i];
        if (scene.m_meshes[i].has_texcoords) scene.m_meshes[i].texcoords = tmp_texcoords[i];
    }
    delete tmp_triangles;
    delete tmp_nodes;
    delete tmp_indices;
    delete tmp_texcoords;
}

SceneRef::~SceneRef()
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
