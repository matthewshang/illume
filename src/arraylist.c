#include "arraylist.h"

#include <stdlib.h>

ArrayList* arraylist_new(int start_size)
{
	ArrayList* array = (ArrayList *) calloc(1, sizeof(ArrayList));
	if (!array)
	{
		return NULL;
	}
	array->length = 0;
	array->size = start_size;
	array->data = calloc(start_size, sizeof(void *));
	return array;
}

void arraylist_free(ArrayList* array)
{
	if (array)
	{
		if (array->data)
		{
			free(array->data);
		}
		
		free(array);
	}
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

void* arraylist_get(ArrayList* array, int index)
{
	if (array && index < array->length)
	{
		return array->data[index];
	}
	else
	{
		return NULL;
	}
}