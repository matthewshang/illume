#ifndef _BITMAP_
#define _BITMAP_

#include <stdint.h>

typedef struct
{
	size_t red;
	size_t green;
	size_t blue;
} 
Pixel;

typedef struct
{
	Pixel* pixels;
	size_t width;
	size_t height;
} 
Bitmap;

#ifdef __cplusplus
extern "C" {
#endif

Bitmap*  bitmap_new          (size_t width, size_t height);
void     bitmap_free         (Bitmap* bitmap);
void     bitmap_set_pixel    (Bitmap* bitmap, int x, int y, int red, int green, int blue);
Pixel*   bitmap_get_pixel    (Bitmap* bitmap, int x, int y);
int      bitmap_save_to_png  (Bitmap* bitmap, char* path);

#ifdef __cplusplus
}
#endif

#endif