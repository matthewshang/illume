#pragma once

#include "../intellisense.h"

struct Vec2f
{
    float x, y;

    __device__ __host__ Vec2f() : x(0.0f), y(0.0f) {};
    __device__ __host__ Vec2f(float _x, float _y) : x(_x), y(_y) {};

    __device__ __host__ 
    Vec2f operator+(Vec2f& v)
    {
        return Vec2f(x + v.x, y + v.y);
    }

    __device__ __host__
    Vec2f operator*(float s)
    {
        return Vec2f(x * s, y * s);
    }

    __device__ __host__
    Vec2f operator*(Vec2f& v)
    {
        return Vec2f(x * v.x, y * v.y);
    }
};