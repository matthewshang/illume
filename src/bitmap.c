#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <png.h> // libpng

#include "bitmap.h"

Bitmap* bitmap_new(size_t width, size_t height)
{
	Bitmap* bitmap = (Bitmap *) calloc(sizeof(Bitmap), 1);
    if (!bitmap)
    {
        return NULL;
    }
	bitmap->width = width;
	bitmap->height = height;
	bitmap->pixels = 
        (Pixel *) calloc(sizeof(Pixel), width * height);
    if (!bitmap->pixels)
    {
        return NULL;
    }
	return bitmap;
}

void bitmap_free(Bitmap* bitmap)
{
    if (bitmap)
    {
        if (bitmap->pixels) 
        {
            free(bitmap->pixels);
        }

        free(bitmap);
    }
}

static int pixel_position_valid(Bitmap* bitmap, int x, int y)
{
    return (y * bitmap->width + x) < (bitmap->width * bitmap->height);
}

void bitmap_set_pixel(Bitmap* bitmap, int x, int y, int red, int green, int blue)
{
	Pixel* pixel = bitmap_get_pixel(bitmap, x, y);
    if (pixel)
    {
    	pixel->red = red;
    	pixel->green = green;
    	pixel->blue = blue;
    }
}

Pixel* bitmap_get_pixel(Bitmap* bitmap, int x, int y)
{
    if (pixel_position_valid(bitmap, x, y))
    {
        return (Pixel *) (bitmap->pixels + bitmap->width * y + x);
    }

    return NULL;
}

// Adapted from http://www.lemoda.net/c/write-png/
int bitmap_save_to_png(Bitmap* bitmap, char* path)
{
    FILE* fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    size_t x, y;
    png_byte** row_pointers = NULL;

    int status = -1;
    int pixel_size = 3;
    int depth = 8;
    
    fp = fopen(path, "wb");
    if (!fp) 
    {
    	printf("pngio: could not save at %s\n", path);
    	printf("errno = %d\n", errno);
        goto fopen_failed;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) 
    {
        goto png_create_write_struct_failed;
    }
    
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) 
    {
        goto png_create_info_struct_failed;
    }

    if (setjmp(png_jmpbuf(png_ptr))) 
    {
    	printf("PNG save at %s failed\n", path);
        goto png_failure;
    }

    png_set_IHDR(png_ptr,
                 info_ptr,
                 bitmap->width,
                 bitmap->height,
                 depth,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    row_pointers = png_malloc(png_ptr, bitmap->height * sizeof(png_byte *));
    for (y = 0; y < bitmap->height; ++y) 
    {
        png_byte* row = 
            png_malloc(png_ptr, sizeof(uint8_t) * bitmap->width * pixel_size);
        row_pointers[y] = row;
        for (x = 0; x < bitmap->width; ++x) 
        {
            Pixel* pixel = bitmap_get_pixel(bitmap, x, y);
            *row++ = pixel->red;
            *row++ = pixel->green;
            *row++ = pixel->blue;
        }
    }

    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    status = 0;
    
    for (y = 0; y < bitmap->height; y++) 
    {
        png_free(png_ptr, row_pointers[y]);
    }
    png_free(png_ptr, row_pointers);


    
 png_failure:
 png_create_info_struct_failed:
    png_destroy_write_struct(&png_ptr, &info_ptr);
 png_create_write_struct_failed:
    fclose(fp);
 fopen_failed:
    return status;
}