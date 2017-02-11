#pragma once

#include "rapidjson/document.h"

#include "math/vector3.h"
#include "math/vector2.h"

namespace JsonUtils
{
	void read_and_parse_json(const char* path, rapidjson::Document& ret);

	void object_from_json(rapidjson::Value& json, int& ret);
	void object_from_json(rapidjson::Value& json, float& ret);
	void object_from_json(rapidjson::Value& json, Vector3& ret);
    void object_from_json(rapidjson::Value& json, Vec2f& ret);
	void object_from_json(rapidjson::Value& json, std::string& ret);
	void object_from_json(rapidjson::Value& json, bool& ret);

	template<typename T>
	inline void from_json(rapidjson::Value& json, const char* name, T& object)
	{
		auto m = json.FindMember(name);
		if (m != json.MemberEnd())
		{
			object_from_json(m->value, object);
		}
		else
		{
			printf("from_json: cannot find property %s\n", name);
		}
	}

	template<typename T>
	inline void from_json(rapidjson::Value& json, const char* name, T& object, T default)
	{
		auto m = json.FindMember(name);
		if (m != json.MemberEnd())
		{
			object_from_json(m->value, object);
		}
		else
		{
			printf("from_json: cannot find property %s, defaulting\n", name);
			object = default;
		}
	}
}