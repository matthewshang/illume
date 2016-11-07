#include "hit.h"

__device__
void hit_set(Hit* hit, float d, Vector3 normal, Material m)
{
    hit->is_intersect = 1;
	hit->d = d;
	hit->normal = normal;
	hit->m = m;
}

__device__
void hit_set_no_intersect(Hit* hit)
{
	hit->is_intersect = 0;
}