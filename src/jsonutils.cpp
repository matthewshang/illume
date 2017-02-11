#include "jsonutils.h"

#include <fstream>
#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "math/vector3.h"

void JsonUtils::read_and_parse_json(const char* path, rapidjson::Document& ret)
{
	std::ifstream t(path);
	std::stringstream buffer;
	buffer << t.rdbuf();
	ret.Parse(buffer.str().c_str());
	if (ret.HasParseError())
	{
		printf("Error parsing JSON(offset %u): %s\n", (unsigned) ret.GetErrorOffset(), rapidjson::GetParseError_En(ret.GetParseError()));
	}
}

void JsonUtils::object_from_json(rapidjson::Value& json, int& ret)
{
	ret = json.GetInt();
}

void JsonUtils::object_from_json(rapidjson::Value& json, float& ret)
{
	ret = json.GetFloat();
}

void JsonUtils::object_from_json(rapidjson::Value& json, Vector3& ret)
{
	ret = vector3_create(json[0].GetFloat(), json[1].GetFloat(), json[2].GetFloat());
}

void JsonUtils::object_from_json(rapidjson::Value& json, Vec2f& ret)
{
    ret = Vec2f(json[0].GetFloat(), json[1].GetFloat());
}

void JsonUtils::object_from_json(rapidjson::Value& json, std::string& ret)
{
	ret = json.GetString();
}

void JsonUtils::object_from_json(rapidjson::Value& json, bool& ret)
{
	ret = json.GetBool();
}
