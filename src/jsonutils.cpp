#include "jsonutils.h"

#include "math/vector3.h"

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

void JsonUtils::object_from_json(rapidjson::Value& json, std::string& ret)
{
	ret = json.GetString();
}
