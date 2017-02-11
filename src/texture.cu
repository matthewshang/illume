#include "texture.h"

#include "jsonutils.h"

Texture texture_from_json(rapidjson::Value& json)
{
    if (json.IsArray())
    {
        return texture_constant(vector3_create(json[0].GetFloat(), json[1].GetFloat(), json[2].GetFloat()));
    }
    std::string type;
    JsonUtils::from_json(json, "type", type);

    if (type == "constant")
    {
        Vector3 color;
        JsonUtils::from_json(json, "color", color, vector3_create(0, 0, 0));
        return texture_constant(color);
    }
    else if (type == "checkerboard")
    {
        Vector3 oncolor, offcolor;
        Vec2f scale;
        JsonUtils::from_json(json, "on_color", oncolor);
        JsonUtils::from_json(json, "off_color", offcolor);
        JsonUtils::from_json(json, "scale", scale, Vec2f(1, 1));
        return texture_checkerboard(oncolor, offcolor, scale);
    }
    else
    {
        printf("texture_from_json: invalid texture type %s\n", type.c_str());
        return texture_constant(vector3_create(0, 0, 0));
    }
}

Texture texture_constant(Vector3 c)
{
    Texture t;
    t.type = TextureType::CONSTANT;
    t.constant.c = c;
    return t;
}

Texture texture_checkerboard(Vector3 on, Vector3 off, Vec2f scale)
{
    Texture t;
    t.type = TextureType::CHECKERBOARD;
    t.checkerboard.on = on;
    t.checkerboard.off = off;
    t.checkerboard.scale = scale;
    return t;
}

__device__
Vector3 Texture::eval(Vec2f uv)
{
    switch (type)
    {
    case TextureType::CONSTANT:
    {
        return constant.c;
    }

    case TextureType::CHECKERBOARD:
    {
        int x = (int) checkerboard.scale.x * uv.x;
        int y = (int) checkerboard.scale.y * uv.y;
        bool on = (x + y) % 2 == 0;
        return on ? checkerboard.on : checkerboard.off;
    }
    }
}
