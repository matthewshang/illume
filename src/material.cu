#include "material.h"

#include "fresnel.h"
#include "jsonutils.h"
#include "conductor.h"

Material material_from_json(rapidjson::Value& json, Medium m, std::vector<Texture>& texture_cache)
{
	std::string type;
	JsonUtils::from_json(json, "type", type);

    Texture albedo, roughness;
    auto albedo_ref = json.FindMember("albedo");
    auto roughness_ref = json.FindMember("roughness");

    if (albedo_ref != json.MemberEnd())
    {
        albedo = texture_from_json(albedo_ref->value);
        texture_cache.push_back(albedo);
    }
    else
    {
        albedo = texture_constant(vector3_create(0, 0, 0));
    }

    if (roughness_ref != json.MemberEnd())
    {
        roughness = texture_from_json(roughness_ref->value);
        texture_cache.push_back(roughness);
    }
    else
    {
        roughness = texture_constant(vector3_create(1e-4, 1e-4, 1e-4));
    }
	if (type == "emissive")
	{
		return material_emissive(albedo);
	}
	else if (type == "lambert")
	{
		return material_diffuse(albedo);
	}
	else if (type == "reflective")
	{
		return material_specular(albedo);
	}
	else if (type == "refractive")
	{
		float ior;
		JsonUtils::from_json(json, "ior", ior);
		return material_refractive(albedo, ior, m);
	}
	else if (type == "roughreflective")
	{
        float ior;
		JsonUtils::from_json(json, "ior",       ior);
		return material_roughreflec(albedo, ior, roughness);
	}
	else if (type == "roughrefractive")
	{
		float ior;
		JsonUtils::from_json(json, "ior",       ior);
		return material_roughrefrac(albedo, ior, roughness, m);
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
	else if (type == "roughconductor")
	{
		std::string material;
		JsonUtils::from_json(json, "material",  material);
		Vector3 eta, k;
		if (!Conductor::get(material, eta, k))
		{
			printf("material_from_json: invalid conductor material %s, defaulting to copper\n", material.c_str());
			Conductor::get("Cu", eta, k);
		}
		return material_roughconductor(eta, k, roughness);
	}
    else if (type == "plastic")
    {
        float ior;
        JsonUtils::from_json(json, "ior", ior);
        return material_plastic(albedo, ior);
    }
	printf("material_from_json: invalid material type %s\n", type.c_str());
	return material_diffuse(texture_constant(vector3_create(0, 0, 0)));
}

Material material_base(Texture albedo, int type)
{
	Material material;
	material.albedo = albedo;
	material.type = type;
	material.ior = 0.f;
    material.diffuse_fresnel = 0.0f;
	material.roughness = texture_constant(vector3_create(1e-4, 1e-4, 1e-4));
	material.medium = medium_air();
    material.eta = vector3_create(0, 0, 0);
	material.k = vector3_create(0, 0, 0);
	return material;
}

Material material_emissive(Texture emission)
{
	return material_base(emission, MATERIAL_EMISSIVE);
}

Material material_diffuse(Texture albedo)
{
	return material_base(albedo, MATERIAL_DIFFUSE);
}

Material material_specular(Texture s)
{
	return material_base(s, MATERIAL_SPECULAR);
}

Material material_refractive(Texture r, float ior, Medium medium)
{
	Material material = material_base(r, MATERIAL_REFRACTIVE);
	material.ior = ior;
	material.medium = medium;
    printf("refractive: %f %f\n", material.albedo.constant.c.x, material.medium.absorption.x);
	return material;
}

Material material_roughreflec(Texture r, float ior, Texture roughness)
{
	Material material = material_base(r, MATERIAL_ROUGHREFLECTIVE);
	material.ior = ior;
	material.roughness = roughness;
	return material;
}

Material material_roughrefrac(Texture r, float ior, Texture roughness, Medium m)
{
	Material material = material_base(r, MATERIAL_ROUGHREFRACTIVE);
	material.ior = ior;
	material.roughness = roughness;
	material.medium = m;
	return material;
}

Material material_conductor(Vector3 eta, Vector3 k)
{
	Material material = material_base(texture_constant(vector3_create(0, 0, 0)), MATERIAL_CONDUCTOR);
    material.eta = eta;
	material.k = k;
    printf("conductor: %f %f\n", material.eta.x, material.k.x);
	return material;
}

Material material_roughconductor(Vector3 eta, Vector3 k, Texture roughness)
{
	Material material = material_base(texture_constant(vector3_create(0, 0, 0)), MATERIAL_ROUGHCONDUCTOR);
    material.eta = eta;
	material.k = k;
	material.roughness = roughness;
	return material;
}

Material material_plastic(Texture specular, float ior)
{
    Material material = material_base(specular, MATERIAL_PLASTIC);
    material.ior = ior;
    material.diffuse_fresnel = Fresnel::diffuse_fresnel(1.0f / ior, 1000000);
    return material;
}