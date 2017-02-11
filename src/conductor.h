#pragma once

#include <string>

#include "math/vector3.h"

namespace Conductor
{
	struct ConductorEntry
	{
		std::string name;
		Vector3 eta;
		Vector3 k;
	};

	// refractiveindex.info/?shelf=3d&book=metals
	// 700, 510, 440
	static const int conductorsAmount = 5;
	static const ConductorEntry conductors[] = {
		{ "Ag", vector3_create(0.16761f, 0.14032f, 0.13529f), vector3_create(4.2868f, 2.8794f, 2.2916f) },
		{ "Al", vector3_create(1.66130f, 0.82373f, 0.58461f), vector3_create(8.0444f, 5.9742f, 5.1875f) },
		{ "Au", vector3_create(0.14311f, 0.37495f, 1.44247f),  vector3_create(3.9831f, 2.3857f, 1.6032f) },
		{ "Cu", vector3_create(0.21249f, 0.97909f, 1.3343f),  vector3_create(4.1005f, 2.3691f, 2.2965f) },
		{ "Pb", vector3_create(1.78000f, 1.74000f, 1.4400f),  vector3_create(3.5700f, 3.3100f, 3.1800f) }
	};

	static bool get(std::string name, Vector3& eta, Vector3& k)
	{
		for (int i = 0; i < conductorsAmount; i++)
		{
			if (conductors[i].name == name)
			{
				eta = conductors[i].eta;
				k = conductors[i].k;
				return true;
			}
		}
		return false;
	}
}