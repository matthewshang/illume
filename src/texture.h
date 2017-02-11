#pragma once

#include "rapidjson/document.h"

#include "math/vector2.h"
#include "math/vector3.h"

enum TextureType
{
    CONSTANT, CHECKERBOARD
};

struct Texture
{
    TextureType type;
    union
    {
        struct
        {
            Vector3 c;
        } constant;

        struct
        {
            Vector3 on, off;
            Vec2f scale;
        } checkerboard;
    };

    Texture()
    {
        type = TextureType::CONSTANT;
        constant.c = vector3_create(0, 0, 0);
    }

    __device__ Vector3 eval(Vec2f uv);
};

Texture texture_from_json(rapidjson::Value& json);

Texture texture_constant(Vector3 c);
Texture texture_checkerboard(Vector3 on, Vector3 off, Vec2f scale);