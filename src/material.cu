#include "material.h"

#include "jsonutils.h"
#include "conductor.h"

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
		JsonUtils::from_json(json, "roughness", roughness, 0.1f);
		return material_roughrefrac(color, ior, roughness, m);
	}
	else if (type == "conductor")
	{
		std::string material;
		JsonUtils::from_json(json, "material", material);
		Vector3 eta, k;
		if (!Conductor::get(material, eta, k))
		{
			printf("material_from_json: invalid conductor material %s, defaulting to copper\n", material.c_str());
			Conductor::get("Cu", eta, k);
		}
		return material_conductor(eta, k);
	}
	else if (type == "rough_conductor")
	{
		std::string material;
		float roughness;
		JsonUtils::from_json(json, "material",  material);
		JsonUtils::from_json(json, "roughness", roughness, 0.1f);
		Vector3 eta, k;
		if (!Conductor::get(material, eta, k))
		{
			printf("material_from_json: invalid conductor material %s, defaulting to copper\n", material.c_str());
			Conductor::get("Cu", eta, k);
		}
		return material_roughconductor(eta, k, roughness);
	}
	printf("material_from_json: invalid material type %s\n", type.c_str());
	return material_diffuse(vector3_create(0, 0, 0));
}

Material material_base(Vector3 c, int type)
{
	Material material;
	material.c = c;
	material.type = type;
	material.ior = 0.f;
	material.roughness = 0.f;
	material.medium = medium_air();
	material.k = vector3_create(0, 0, 0);
	return material;
}

Material material_emissive(Vector3 e)
{
	return material_base(e, MATERIAL_EMISSIVE);
}

Material material_diffuse(Vector3 d)
{
	return material_base(d, MATERIAL_DIFFUSE);
}

Material material_specular(Vector3 s)
{
	return material_base(s, MATERIAL_SPECULAR);
}

Material material_refractive(Vector3 r, float ior, Medium medium)
{
	Material material = material_base(r, MATERIAL_REFRACTIVE);
	material.ior = ior;
	material.medium = medium;
	return material;
}

Material material_cooktorrance(Vector3 r, float ior, float roughness)
{
	Material material = material_base(r, MATERIAL_COOKTORRANCE);
	material.ior = ior;
	material.roughness = roughness;
	return material;
}

Material material_roughrefrac(Vector3 r, float ior, float roughness, Medium m)
{
	Material material = material_base(r, MATERIAL_ROUGHREFRACTIVE);
	material.ior = ior;
	material.roughness = roughness;
	material.medium = m;
	return material;
}

Material material_conductor(Vector3 eta, Vector3 k)
{
	Material material = material_base(eta, MATERIAL_CONDUCTOR);
	material.k = k;
	return material;
}

Material material_roughconductor(Vector3 eta, Vector3 k, float roughness)
{
	Material material = material_base(eta, MATERIAL_ROUGHCONDUCTOR);
	material.k = k;
	material.roughness = roughness;
	return material;
}