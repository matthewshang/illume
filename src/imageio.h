#pragma once

#include <string>

class ImageIO
{
public:
    static void load_ldr(std::string path, unsigned char** data, int* width, int* height, int* channels);
    static void load_hdr(std::string path, float** data, int* width, int* height, int* channels);
};