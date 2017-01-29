#include "material.h"

#include "jsonutils.h"

Material material_from_json(rapidjson::Value& json, Medium m)
{
	std::string type;
	JsonUtils::from_json(json, "type", type);

	Vector3 color;
	JsonUtils::from_json(json, "color", color);
	if (type == "emissive")
	{
		return material_emissive(color);
	}
	else if (type == "diffuse")
	{
		return material_diffuse(color);
	}
	else if (type == "specular")
	{
		return material_specular(color);
	}
	else if (type == "refractive")
	{
		float ior;
		JsonUtils::from_json(json, "ior", ior);
		return material_refractive(color, ior, m);
	}
	else if (type == "cooktorrance")
	{
		float ior, roughness;
		JsonUtils::from_json(json, "ior",       ior);
		JsonUtils::from_json(json, "roughness", roughness);
		return material_cooktorrance(color, ior, roughness);
	}
	else if (type == "rough_glass")
	{
		float ior, roughness;
		JsonUtils::from_json(json, "ior",       ior);
		JsonUtils::from_json(json, "roughness", roughness);
		return material_roughrefrac(color, ior, roughness, m);
	}
	printf("material_from_json: invalid material type %s\n", type.c_str());
	return material_diffuse(vector3_create(0, 0, 0));
}

Material material_emissive(Vector3 e)
{
	Material material;
	material.c = e;
	material.type = MATERIAL_EMISSIVE;
	material.ior = 0.f;
	material.roughness = 0.f;
	material.medium = medium_air();
	return material;
}

Material material_diffuse(Vector3 d)
{
	Material material;
	material.c = d;
	material.type = MATERIAL_DIFFUSE;
	material.ior = 0.f;
	material.roughness = 0.f;
	material.medium = medium_air();
	return material;
}

Material material_specular(Vector3 s)
{
	Material material;
	material.c = s;
	material.type = MATERIAL_SPECULAR;
	material.ior = 0.f;
	material.roughness = 0.f;
	material.medium = medium_air();
	return material;
}

Material material_refractive(Vector3 r, float ior, Medium medium)
{
	Material material;
	material.c = r;
	material.type = MATERIAL_REFRACTIVE;
	material.ior = ior;
	material.roughness = 0.f;
	material.medium = medium;
	return material;
}

Material material_cooktorrance(Vector3 r, float ior, float roughness)
{
	Material material;
	material.c = r;
	material.type = MATERIAL_COOKTORRANCE;
	material.ior = ior;
	material.roughness = roughness;
	material.medium = medium_air();
	return material;
}

Material material_roughrefrac(Vector3 r, float ior, float roughness, Medium m)
{
	Material material;
	material.c = r;
	material.type = MATERIAL_ROUGHREFRACTIVE;
	material.ior = ior;
	material.roughness = roughness;
	material.medium = m;
	return material;
}
