#ifndef _INTERSECTION_
#define _INTERSECTION_

typedef struct
{
	int is_intersect;
}
Intersection;

__device__  Intersection  intersection_create  (int is_intersect);

#endif