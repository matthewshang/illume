#pragma once

#include "../intellisense.h"

struct Vec2f
{
    float x, y;

    __device__ __host__ Vec2f() : x(0.0f), y(0.0f) {};
    __device__ __host__ Vec2f(float _x, float _y) : x(_x), y(_y) {};
};