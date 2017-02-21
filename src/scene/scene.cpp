#include "scene.h"

#include <vector>
#include <unordered_map>

#include "rapidjson/document.h"

#include "../arraylist.h"
#include "../jsonutils.h"

HostScene::HostScene(rapidjson::Document& json)
{
    std::unordered_map<std::string, Medium> medium_map;
    std::unordered_map<std::string, Material> mat_map;
    std::unordered_map<std::string, int> mesh_index_map;

    auto envmap = json.FindMember("environment_map");
    auto camera = json.FindMember("camera");
    auto mediums = json.FindMember("mediums");
    auto materials = json.FindMember("materials");
    auto meshes_loc = json.FindMember("meshes");
    auto primitives = json.FindMember("primitives");

    if (envmap != json.MemberEnd())
    {
        m_environment = texture_from_json(envmap->value);
        m_textures.push_back(m_environment);
    }

    if (camera != json.MemberEnd())
    {
        m_camera = camera_from_json(camera->value);
    }

    if (mediums != json.MemberEnd())
    {
        for (auto& itr : mediums->value.GetArray())
        {
            std::string name;
            JsonUtils::from_json(itr, "name", name);
            medium_map.emplace(std::make_pair(name, medium_from_json(itr)));
        }
    }

    if (materials != json.MemberEnd())
    {
        for (auto& itr : materials->value.GetArray())
        {
            std::string name, medium_name;
            JsonUtils::from_json(itr, "name", name);
            JsonUtils::from_json(itr, "medium", medium_name);
            Medium medium;
            auto medium_ref = medium_map.find(medium_name);
            if (medium_ref == medium_map.end())
            {
                medium = medium_air();
            }
            else
            {
                medium = medium_ref->second;
            }

            mat_map.emplace(std::make_pair(name, material_from_json(itr, medium, m_textures)));
        }
    }

    if (meshes_loc != json.MemberEnd())
    {
        for (auto& itr : meshes_loc->value.GetArray())
        {
            std::string name;
            JsonUtils::from_json(itr, "name", name);

            mesh_index_map.emplace(std::make_pair(name, m_meshes.size()));
            m_meshes.push_back(mesh_from_json(itr));
        }
    }

    if (primitives != json.MemberEnd())
    {
        for (auto& itr : primitives->value.GetArray())
        {
            std::string type, mat_name;
            JsonUtils::from_json(itr, "material", mat_name);
            JsonUtils::from_json(itr, "type", type);
            if (type == "sphere")
            {
                m_spheres.push_back(sphere_from_json(itr, mat_map[mat_name]));
            }
            else if (type == "mesh_instance")
            {
                std::string mesh_name;
                JsonUtils::from_json(itr, "mesh", mesh_name);
                int mesh_index = mesh_index_map[mesh_name];
                m_instances.push_back(mesh_instance_from_json(itr, mat_map[mat_name], mesh_index, &m_meshes[mesh_index]));
            }
        }
    }
}

HostScene::~HostScene()
{
    for (Texture& t : m_textures)
    {
        t.destroy();
    }

    for (Mesh& m : m_meshes)
    {
        mesh_destroy(m);
    }
}