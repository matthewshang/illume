#include "hit.h"

__device__
void hit_set(Hit* hit, float d, Vector3 normal, Material* m, Vec2f uv)
{
    hit->is_intersect = true;
	hit->d = d;
	hit->normal = normal;
	hit->m = m;
    hit->uv = uv;
}

__device__
void hit_set_no_intersect(Hit* hit)
{
	hit->is_intersect = false;
}