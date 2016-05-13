#include "kdtree.h"

static KDTreeNode* new_node(AABB bounds)
{
	KDTreeNode* node = (KDTreeNode *) calloc(1, sizeof(KDTreeNode));
	node->bounds = bounds;
	return node;
}

static void init_leaf(KDTreeNode* node, int prim_amount)
{
	node->left_index = -1;
	node->right_index = -1;
	node->prim_start = -1;
	node->prim_amount = prim_amount;
}

static void init_node(KDTreeNode* node, int left_index, int right_index)
{
	node->left_index = left_index;
	node->right_index = right_index;
	node->prim_start = -1;
	node->prim_amount = 0;
}

static int node_is_leaf(KDTreeNode* node)
{
	return node->left_index == -1 && node->right_index == -1;
}

static float get_bounds(AABB* left, AABB* right, int axis, AABB current)
{
	Vector3 min = current.min;
	Vector3 max = current.max;
	float split_value;
	if (axis == X_AXIS)
	{
		split_value = (min.x + max.x) / 2;
		*left = aabb_from_vertices(min, vector3_create(split_value, max.y, max.z));
		*right = aabb_from_vertices(vector3_create(split_value, min.y, min.z), max);
	}
	else if (axis == Y_AXIS)
	{
		split_value = (min.y + max.y) / 2;
		*left = aabb_from_vertices(min, vector3_create(split_value, max.y, max.z));
		*right = aabb_from_vertices(vector3_create(split_value, min.y, min.z), max);
	}
	else
	{
		split_value = (min.z + max.z) / 2;
		*left = aabb_from_vertices(min, vector3_create(split_value, max.y, max.z));
		*right = aabb_from_vertices(vector3_create(split_value, min.y, min.z), max);
	}
	return split_value;
}

static void build(KDTreeNode* node, int index, AABB* prims, int prim_amount, int depth, int max_prims, 
				  ArrayList* nodes, ArrayList* node_prims, int axis)
{
	if (node->prim_amount > max_prims && depth != 0)
	{
		int split_axis = (axis + 1) % 3;
		AABB left_bounds;
		AABB right_bounds;
		float split_value = get_bounds(&left_bounds, &right_bounds, split_axis, node->bounds);

		int* left_prims = (int *) calloc(node->prim_amount, sizeof(int));
		int left_prims_amount = 0;
		int* right_prims = (int *) calloc(node->prim_amount, sizeof(int));
		int right_prims_amount = 0;
		int* current_prims = (int *) arraylist_get(node_prims, index);
		for (int i = 0; i < node->prim_amount; i++)
		{
			AABB current = prims[current_prims[i]];
			if (aabb_aabb_intersect(left_bounds, current))
			{
				left_prims[i] = i;
				left_prims_amount++;
			}
			else
			{
				left_prims[i] = -1;
			}
			if (aabb_aabb_intersect(right_bounds, current))
			{
				right_prims[i] = i;
				right_prims_amount++;
			}
			else
			{
				right_prims[i] = -1;
			}
		}

		int* new_left_prims = (int *) calloc(left_prims_amount, sizeof(int));
		int* new_right_prims = (int *) calloc(right_prims_amount, sizeof(int));
		left_prims_amount = 0;
		right_prims_amount = 0;
		for (int i = 0; i < node->prim_amount; i++)
		{
			if (left_prims[i] != -1)
			{
				new_left_prims[left_prims_amount] = left_prims[i];
				left_prims_amount++;
			}
			if (right_prims[i] != -1)
			{
				new_right_prims[right_prims_amount] = right_prims[i];
				right_prims_amount++;
			}
		}
		free(left_prims);
		free(right_prims);

		int left_index = nodes->length;
		KDTreeNode* left = new_node(left_bounds);
		init_leaf(left, left_prims_amount);
		arraylist_add(nodes, left);
		arraylist_add(node_prims, new_left_prims);
		build(left, left_index, prims, prim_amount, depth - 1, max_prims, nodes, node_prims, axis);
		
		int right_index = nodes->length;
		KDTreeNode* right = new_node(right_bounds);
		init_leaf(right, right_prims_amount);
		arraylist_add(nodes, right);
		arraylist_add(node_prims, new_right_prims);
		build(right, right_index, prims, prim_amount, depth - 1, max_prims, nodes, node_prims, axis);

		init_node(node, left_index, right_index);
	}
}

KDTree kdtree_build(AABB* prims, int prim_amount, AABB bounds, int max_depth, int max_prims)
{
	ArrayList* nodes = arraylist_new(1);
	ArrayList* node_prims = arraylist_new(1);
	KDTreeNode* root = new_node(bounds);
	init_leaf(root, prim_amount);
	arraylist_add(nodes, root);
	int* root_prims = (int *) calloc(prim_amount, sizeof(int));
	for (int i = 0; i < prim_amount; i++)
	{
		root_prims[i] = i;
	}
	arraylist_add(node_prims, root_prims);
	build(root, 0, prims, prim_amount, max_depth, max_prims, nodes, node_prims, -1);

	KDTree tree;
	tree.node_amount = nodes->length;
	tree.nodes = (KDTreeNode *) calloc(tree.node_amount, sizeof(KDTreeNode));
	int total_prims = 0;
	for (int i = 0; i < nodes->length; i++)
	{
		KDTreeNode* node = (KDTreeNode *) arraylist_get(nodes, i);
		if (node_is_leaf(node))
		{
			node->prim_start = total_prims;
			total_prims += node->prim_amount;
		}
		tree.nodes[i] = *node;
		free(node);
	}
	arraylist_free(nodes);
	tree.node_prims = (int *) calloc(total_prims, sizeof(int));
	for (int i = 0; i < tree.node_amount; i++)
	{
		KDTreeNode* node = &tree.nodes[i];
		int* current_prims = (int *) arraylist_get(node_prims, i);
		if (node_is_leaf(node))
		{
			for (int j = 0; j < node->prim_amount; j++)
			{
				tree.node_prims[node->prim_start + j] = current_prims[j];
			}
		}
		if (current_prims)
		{
			free(current_prims);
		}
	}
	arraylist_free(node_prims);
	return tree;
}

void kdtree_free(KDTree* tree)
{
	if (tree)
	{
		if (tree->nodes)
		{
			free(tree->nodes);
		}
		if (tree->node_prims)
		{
			free(tree->node_prims);
		}
	}
}