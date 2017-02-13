#include "imageio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void ImageIO::load_image(std::string path, void** data, int* width, int* height, int* channels)
{
    *data = stbi_load(path.c_str(), width, height, channels, 4);
    printf("%d\n", *channels);
    if (!*data)
    {
        printf("ImageIO::load_image: could not load image at %s\n", path.c_str());
    }
}