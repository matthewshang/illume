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

ArrayList   arraylist_new   (int size);
void        arraylist_add   (ArrayList* array, void* item);
void        arraylist_free  (ArrayList* array);

#endif