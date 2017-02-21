#include "texture.h"

#include "error_check.h"
#include "imageio.h"
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
    else if (type == "bitmap")
    {
        std::string path;
        Vec2f scale;
        JsonUtils::from_json(json, "file", path, std::string(""));
        JsonUtils::from_json(json, "scale", scale, Vec2f(1, 1));

        if (path == "")
        {
            printf("texture_from_json: filepath not specified\n");
            return texture_constant(vector3_create(0, 0, 0));
        }
       
        if (path.substr(path.length() - 3) == "hdr")
        {
            float* data;
            int width, height, channels;
            ImageIO::load_hdr(path, &data, &width, &height, &channels);
            if (!data) return texture_constant(vector3_create(0, 0, 0));
            return texture_bitmap<float4>(data, width, height, scale, true);
        }
        else
        {
            unsigned char* data;
            int width, height, channels;
            ImageIO::load_ldr(path, &data, &width, &height, &channels);
            if (!data) return texture_constant(vector3_create(0, 0, 0));
            return texture_bitmap<uchar4>(data, width, height, scale, false);
        }
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

template <typename T>
Texture texture_bitmap(void* data, int width, int height, Vec2f scale, bool is_hdr)
{
    Texture t;
    t.type = TextureType::BITMAP;
    //size_t size = width * height * channels * item_size;
    size_t size = width * height * sizeof(T);
    //printf("%d %d\n", width, height);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    HANDLE_ERROR( cudaMallocArray(&t.bitmap.devBuffer, &channelDesc, width, height) );
    HANDLE_ERROR( cudaMemcpyToArray(t.bitmap.devBuffer, 0, 0, data, size, cudaMemcpyHostToDevice) );

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = t.bitmap.devBuffer;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = is_hdr ? cudaFilterModeLinear : cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;

    texDesc.normalizedCoords = 1;

    t.bitmap.texObj = 0;
    HANDLE_ERROR( cudaCreateTextureObject(&t.bitmap.texObj, &resDesc, &texDesc, NULL) );
    
    t.bitmap.is_hdr = is_hdr;
    t.bitmap.width = width;
    t.bitmap.height = height;
    t.bitmap.scale = scale;

    free(data);
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
    case TextureType::BITMAP:
    {
        uv = uv * bitmap.scale;
        if (bitmap.is_hdr)
        {
            float4 tex = tex2D<float4>(bitmap.texObj, uv.x, uv.y);
            return vector3_create(tex.x, tex.y, tex.z);
        }
        else
        {
            uchar4 tex = tex2D<uchar4>(bitmap.texObj, uv.x, uv.y);
            return vector3_create((float)tex.x / 255.0f, (float)tex.y / 255.0f, (float)tex.z / 255.0f);
        }
    }
    }
    return vector3_create(0, 0, 0);
}

void Texture::destroy()
{
    if (type == TextureType::BITMAP)
    {
        HANDLE_ERROR( cudaDestroyTextureObject(bitmap.texObj) );
        HANDLE_ERROR( cudaFreeArray(bitmap.devBuffer) );
    }
}
