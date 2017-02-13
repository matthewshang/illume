#pragma once

#include <string>

class ImageIO
{
public:
    static void load_image(std::string path, void** data, int* width, int* height, int* channels);
};