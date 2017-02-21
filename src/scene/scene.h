#pragma once

#include "rapidjson/document.h"

#include "../primitives/sphere.h"
#include "../primitives/mesh.h"
#include "../primitives/mesh_instance.h"
#include "../camera.h"
#include "../math/vector3.h"

class HostScene
{
friend class SceneRef;
public:
    HostScene(rapidjson::Document& json);
    ~HostScene();

    inline Camera& get_camera() { return m_camera; }
    inline std::vector<Mesh>& get_meshes() { return m_meshes; }
private:
    std::vector<Mesh> m_meshes;
    std::vector<Sphere> m_spheres;
    std::vector<MeshInstance> m_instances;
    std::vector<Texture> m_textures;

    Texture m_environment;
    Camera m_camera;
};

struct DeviceScene
{
	Sphere* spheres;
	int sphere_amount;
	Mesh* meshes;
	int mesh_amount;
	MeshInstance* instances;
	int instance_amount;
	Camera camera;
	Texture envmap;
};