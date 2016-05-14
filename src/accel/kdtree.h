#ifndef _KDTREE_
#define _KDTREE_

#include "../math/aabb.h"
#include "../math/constants.h"
#include "../math/vector3.h"
#include "../arraylist.h"

const int X_AXIS = 0;
const int Y_AXIS = 1;
const int Z_AXIS = 2;

const int LEFT_S = 0;
const int RIGHT_S = 1;
const int BOTTOM_S = 2;
const int TOP_S = 3;
const int BACK_S = 4;
const int FRONT_S = 5;

typedef struct
{
	int split_axis;
	float split_value;
	int left_index;
	int right_index;
	int prim_start;
	int prim_amount;
	int ropes[6];
}
KDTreeNode;

typedef struct
{
	int node_amount;
	KDTreeNode* nodes;
	int total_prims;
	int* node_prims;
}
KDTree;

#ifdef __cplusplus
extern "C" {
#endif

KDTree  kdtree_build  (AABB* prims, int prim_amount, AABB bounds, int depth, int max_prims);
void    kdtree_free   (KDTree* tree);

#ifdef __cplusplus
}
#endif

#endif