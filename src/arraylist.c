#include "arraylist.h"

ArrayList arraylist_new(int start_size)
{
	ArrayList array;
	array.length = 0;
	array.size = start_size;
	array.data = calloc(start_size, sizeof(void *));
	return array;
}

static void arraylist_resize(ArrayList* array)
{
	if (array && array->data)
	{
		array->data = realloc(array->data, array->size * 2 * sizeof(void *));
		if (array->data)
		{
			array->size *= 2;
		}
	}
}

void arraylist_add(ArrayList* array, void* item)
{
	if (array)
	{
		if (array->length == array->size)
		{
			arraylist_resize(array);
		}

		array->data[array->length++] = item;
	}
}

void arraylist_free(ArrayList* array)
{
	if (array)
	{
		free(array->data);
		array->size = 0;
		array->length = 0;
	}
}