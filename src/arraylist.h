#ifndef _ARRAYLIST_
#define _ARRAYLIST_

#include <stdlib.h>

typedef struct
{
	int length;
	int size;
	void** data;
}
ArrayList;

#ifdef __cplusplus
extern "C" {
#endif

ArrayList*   arraylist_new      (int size);
void         arraylist_free     (ArrayList* array);
void         arraylist_add      (ArrayList* array, void* item);
void*        arraylist_get      (ArrayList* array, int index);

#ifdef __cplusplus
}
#endif

#endif