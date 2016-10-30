#include "material.h"

Material material_emissive(Vector3 e)
{
	Material material;
	material.c = e;
	material.type = MATERIAL_EMISSIVE;
	material.ior = 0.f;
	return material;
}

Material material_diffuse(Vector3 d)
{
	Material material;
	material.c = d;
	material.type = MATERIAL_DIFFUSE;
	material.ior = 0.f;
	return material;
}

Material material_specular(Vector3 s)
{
	Material material;
	material.c = s;
	material.type = MATERIAL_SPECULAR;
	material.ior = 0.f;
	return material;
}

Material material_refractive(Vector3 r, float ior)
{
	Material material;
	material.c = r;
	material.type = MATERIAL_REFRACTIVE;
	material.ior = ior;
	return material;
}