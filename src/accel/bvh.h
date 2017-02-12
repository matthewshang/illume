#ifndef _BVH_
#define _BVH_

#include <vector>

#include "../arraylist.h"
#include "../math/aabb.h"

#define BVH_STACK_SIZE 32

typedef struct GPUNode
{
	AABB bounds;
	union
	{
		struct
		{
			unsigned left_node;
			unsigned right_node;
		} node;

		struct
		{
			unsigned tri_amount;
			unsigned tri_start;
		} leaf;
	} u;
} GPUNode;

typedef struct
{
	int* tri_indices;
	int tri_index_amount;
	GPUNode* nodes;
	int node_amount;
} BVH;

#ifdef __cplusplus
extern "C" {
#endif

BVH bvh_create(std::vector<AABB> aabbs, AABB bounds, char* filename, int tris_per_node);
void bvh_free(BVH* bvh);

#ifdef __cplusplus
}
#endif

#endif