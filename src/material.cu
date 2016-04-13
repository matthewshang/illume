#include "material.h"

Material material_emissive(Vector3 e)
{
	Material material;
	material.c = e;
	material.type = MATERIAL_EMISSIVE;
	return material;
}

Material material_diffuse(Vector3 d)
{
	Material material;
	material.c = d;
	material.type = MATERIAL_DIFFUSE;
	return material;
}

Material material_specular(Vector3 s)
{
	Material material;
	material.c = s;
	material.type = MATERIAL_SPECULAR;
	return material;
}