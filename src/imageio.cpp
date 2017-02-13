#include "imageio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void ImageIO::load_ldr(std::string path, unsigned char** data, int* width, int* height, int* channels)
{
    *data = stbi_load(path.c_str(), width, height, channels, 4);
    printf("%d\n", *channels);
    if (!*data)
    {
        printf("ImageIO::load_ldr: could not load image at %s\n", path.c_str());
    }
}

void ImageIO::load_hdr(std::string path, float** data, int* width, int* height, int* channels)
{
    *data = stbi_loadf(path.c_str(), width, height, channels, 4);
    printf("%d\n", *channels);
    if (!*data)
    {
        printf("ImageIO::load_ldr: could not load image at %s\n", path.c_str());
    }
}