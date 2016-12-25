#include "scene.h"

#include <vector>
#include <unordered_map>

#include "rapidjson/document.h"

#include "../arraylist.h"
#include "../jsonutils.h"

void scene_from_json(Scene* scene, rapidjson::Document& json)
{
	JsonUtils::from_json(json, "bgcolor", scene->sky_color);

	std::unordered_map<std::string, Material> mat_map;
	std::unordered_map<std::string, int> mesh_index_map;
	std::vector<Mesh> meshes;
	std::vector<Sphere> spheres;
	std::vector<MeshInstance> instances;

	auto camera     = json.FindMember("camera");
	auto materials  = json.FindMember("materials");
	auto meshes_loc = json.FindMember("meshes");
	auto primitives = json.FindMember("primitives");

	if (camera != json.MemberEnd())
	{
		scene->camera = camera_from_json(camera->value);
	}

	if (materials != json.MemberEnd())
	{
		for (auto& itr : materials->value.GetArray())
		{
			std::string name;
			JsonUtils::from_json(itr, "name", name);
			mat_map.emplace(std::make_pair(name, material_from_json(itr)));
		}
	}

	if (meshes_loc != json.MemberEnd())
	{
		for (auto& itr : meshes_loc->value.GetArray())
		{
			std::string name, file;
			int z_up, items_per_node;
			JsonUtils::from_json(itr, "name",               name);
			JsonUtils::from_json(itr, "file",               file);
			JsonUtils::from_json(itr, "z_up",               z_up);
			JsonUtils::from_json(itr, "bvh_items_per_node", items_per_node);

			mesh_index_map.emplace(std::make_pair(name, meshes.size()));
			meshes.push_back(mesh_create(file.c_str(), z_up, items_per_node));
		}
	}

	if (primitives != json.MemberEnd())
	{
		for (auto& itr : primitives->value.GetArray())
		{
			std::string type, mat_name;
			JsonUtils::from_json(itr, "material", mat_name);
			JsonUtils::from_json(itr, "type",     type);
			if (type == "sphere")
			{
				spheres.push_back(sphere_from_json(itr, mat_map[mat_name]));
			}
			else if (type == "mesh_instance")
			{
				std::string mesh_name;
				JsonUtils::from_json(itr, "mesh", mesh_name);
				int mesh_index = mesh_index_map[mesh_name];
				instances.push_back(mesh_instance_from_json(itr, mat_map[mat_name], mesh_index, &meshes[mesh_index]));
			}
		}

		scene->sphere_amount = spheres.size();
		scene->spheres = (Sphere *) calloc(scene->sphere_amount, sizeof(Sphere));
		std::copy(spheres.begin(), spheres.end(), scene->spheres);

		scene->mesh_amount = meshes.size();
		scene->meshes = (Mesh *) calloc(scene->mesh_amount, sizeof(Mesh));
		std::copy(meshes.begin(), meshes.end(), scene->meshes);

		scene->instance_amount = instances.size();
		scene->instances = (MeshInstance *) calloc(scene->instance_amount, sizeof(MeshInstance));
		std::copy(instances.begin(), instances.end(), scene->instances);
	}
}

Scene* scene_new(rapidjson::Document& json)
{
	Scene* scene = (Scene *) calloc(1, sizeof(Scene));

	scene_from_json(scene, json);

	return scene;
}

void scene_free(Scene* scene)
{
	if (scene)
	{
		if (scene->spheres)
		{
			free(scene->spheres);
		}

		if (scene->instances)
		{
			free(scene->instances);
		}

		if (scene->meshes)
		{
			for (int i = 0; i < scene->mesh_amount; i++)
			{
				if (scene->meshes[i].triangles)
				{
					free(scene->meshes[i].triangles);
				}
				bvh_free(&scene->meshes[i].bvh);
			}
			free(scene->meshes);
		}

		free(scene);
	}
}