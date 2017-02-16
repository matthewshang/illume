#include "tonemapper.h"

#include "jsonutils.h"

Tonemapper::Tonemapper(rapidjson::Value& json)
{
    std::string op;
    JsonUtils::from_json(json, "tonemapper", op, std::string("linear"));
    if (op == "linear")
    {
        m_op = Operator::LINEAR;
    }
    else if (op == "Reinhard")
    {
        m_op = Operator::REINHARD;
    }
    else if (op == "Filmic")
    {
        m_op = Operator::FILMIC;
    }
    else if (op == "Uncharted")
    {
        m_op = Operator::UNCHARTED;
    }
    else
    {
        printf("tonemapper: invalid operator %s\n", op.c_str());
        m_op = Operator::LINEAR;
    }
}

// filmicworlds.com/blog/filmic-tonemapping-operators/
__device__
Vector3 Uncharted2Tonemap(Vector3 x)
{
    const float A = 0.15f, B = 0.50f, C = 0.10f,
        D = 0.20f, E = 0.02f, F = 0.30f;
    return vector3_add(vector3_div(vector3_add(vector3_mul(x, vector3_add(vector3_mul(x, A), C * B)), D * E),
        vector3_add(vector3_mul(x, vector3_add(vector3_mul(x, A), B)), D * F)), -E / F);
}

__device__
Vector3 Tonemapper::eval(Vector3 in)
{
    const float gamma = 1.0f / 2.2f;
    switch (m_op)
    {
    case Operator::LINEAR:
    {
        return vector3_pow(in, gamma);
    }
    case Operator::REINHARD:
    {
        return vector3_pow(vector3_div(in, vector3_add(in, 1.0f)), gamma);
    }
    case Operator::FILMIC:
    {
        Vector3 x = vector3_max(vector3_add(in, -0.004f), 0.0f);
        return vector3_div(vector3_mul(x, vector3_add(vector3_mul(x, 6.2f), 0.5f)),
            vector3_add(vector3_mul(x, vector3_add(vector3_mul(x, 6.2f), 1.7f)), 0.06f));
    }
    case Operator::UNCHARTED:
    {
        const float W = 11.2f;
        const float exposure_bias = 2.0f;
        Vector3 curr = Uncharted2Tonemap(vector3_mul(in, exposure_bias));
        Vector3 white_scale = vector3_div(vector3_create(1.0f, 1.0f, 1.0f), Uncharted2Tonemap(vector3_create(W, W, W)));
        Vector3 color = vector3_mul(curr, white_scale);
        return vector3_pow(color, gamma);
    }
    default:
    {
        return vector3_pow(in, gamma);
    }
    }
}