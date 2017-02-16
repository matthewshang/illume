#pragma once

#include "rapidjson/document.h"

#include "math/vector3.h"

class Tonemapper
{
public:
    enum Operator
    {
        LINEAR, REINHARD, FILMIC, UNCHARTED
    };

    Tonemapper() : m_op(Operator::LINEAR) {};
    Tonemapper(rapidjson::Value& json);

    __device__ Vector3 eval(Vector3 in);

private:
    Operator m_op;
};