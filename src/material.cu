#include "material.h"

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
	material.roughness = roughness * roughness;
	material.medium = medium_air();
	return material;
}
