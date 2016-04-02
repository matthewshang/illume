#include "material.h"

Material material_emissive(Vector3 e)
{
	Material material;
	material.e = e;
	material.d = vector3_create(0, 0, 0);
	return material;
}

Material material_diffuse(Vector3 d)
{
	Material material;
	material.e = vector3_create(0, 0, 0);
	material.d = d;
	return material;
}