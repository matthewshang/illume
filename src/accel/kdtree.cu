#include "kdtree.h"

static KDTreeNode* new_leaf(int prim_amount)
{
	KDTreeNode* node = (KDTreeNode *) calloc(1, sizeof(KDTreeNode));
	node->left_index = -1;
	node->right_index = -1;
	node->prim_start = -1;
	node->prim_amount = prim_amount;
	node->split_axis = -1;
	node->split_value = 0;
	return node;
}

static void init_node(KDTreeNode* node, int left_index, int right_index, 
				      int split_axis, float split_value)
{
	node->left_index = left_index;
	node->right_index = right_index;
	node->prim_start = -1;
	node->prim_amount = 0;
	node->split_axis = split_axis;
	node->split_value = split_value;
}

static int node_is_leaf(KDTreeNode* node)
{
	return node->left_index == -1;
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
		*left = aabb_from_vertices(min, vector3_create(max.x, split_value, max.z));
		*right = aabb_from_vertices(vector3_create(min.x, split_value, min.z), max);
	}
	else
	{
		split_value = (min.z + max.z) / 2;
		*left = aabb_from_vertices(min, vector3_create(max.x, max.y, split_value));
		*right = aabb_from_vertices(vector3_create(min.x, min.y, split_value), max);
	}
	return split_value;
}

static void build(KDTreeNode* node, int index, AABB* prims, int prim_amount, int depth, int max_prims, 
				  ArrayList* nodes, ArrayList* node_prims, int axis, AABB current_bounds)
{
	if (node->prim_amount > max_prims && depth != 0)
	{
		int split_axis = (axis + 1) % 3;
		AABB left_bounds;
		AABB right_bounds;
		float split_value = get_bounds(&left_bounds, &right_bounds, split_axis, current_bounds);

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
		KDTreeNode* left = new_leaf(left_prims_amount);
		arraylist_add(nodes, left);
		arraylist_add(node_prims, new_left_prims);
		build(left, left_index, prims, prim_amount, depth - 1, max_prims, nodes, node_prims, split_axis, left_bounds);
		
		int right_index = nodes->length;
		KDTreeNode* right = new_leaf(right_prims_amount);
		arraylist_add(nodes, right);
		arraylist_add(node_prims, new_right_prims);
		build(right, right_index, prims, prim_amount, depth - 1, max_prims, nodes, node_prims, split_axis, right_bounds);

		init_node(node, left_index, right_index, split_axis, split_value);
	}
}

static void optimize_ropes(KDTreeNode* current_node, int* ropes, KDTreeNode* nodes, AABB bounds)
{
	for (int i = 0; i < 6; i++)
	{
		if (ropes[i] == -1)
		{
			continue;
		}

		KDTreeNode* rope_node = &nodes[ropes[i]];
		while (!node_is_leaf(rope_node))
		{
			rope_node = &nodes[ropes[i]];
			if (i == LEFT_S || i == RIGHT_S)
			{
				if (rope_node->split_axis == X_AXIS)
				{
					if (i == LEFT_S)
					{
						ropes[i] = rope_node->right_index;
					}
					else
					{
						ropes[i] = rope_node->left_index;
					}
				}
				else if (rope_node->split_axis == Y_AXIS)
				{
					if (rope_node->split_value > bounds.min.y - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.y + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}
				}
				else if (rope_node->split_axis == Z_AXIS)
				{
					if (rope_node->split_value > bounds.min.z - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.z + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}
				}
			}
			else if (i == BOTTOM_S || i == TOP_S)
			{
				if (rope_node->split_axis == X_AXIS)
				{
					if (rope_node->split_value > bounds.min.x - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.x + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}	
				}
				else if (rope_node->split_axis == Y_AXIS)
				{
					if (i == BOTTOM_S)
					{
						ropes[i] = rope_node->right_index;
					}
					else
					{
						ropes[i] = rope_node->left_index;
					}
				}
				else if (rope_node->split_axis == Z_AXIS)
				{
					if (rope_node->split_value > bounds.min.z - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.z + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				if (rope_node->split_axis == X_AXIS)
				{
					if (rope_node->split_value > bounds.min.x - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.x + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}
				}
				else if (rope_node->split_axis == Y_AXIS)
				{
					if (rope_node->split_value > bounds.min.y - ILLUME_EPS)
					{
						ropes[i] = rope_node->right_index;
					}
					else if (rope_node->split_value < bounds.max.y + ILLUME_EPS)
					{
						ropes[i] = rope_node->left_index;
					}
					else
					{
						break;
					}
				}
				else if (rope_node->split_axis == Z_AXIS)
				{
					if (i == BACK_S)
					{
						ropes[i] = rope_node->right_index;
					}
					else
					{
						ropes[i] = rope_node->left_index;
					}
				}
			}
		}
	}
}

static void build_ropes(KDTreeNode* current_node, int* ropes, KDTreeNode* nodes, AABB bounds)
{
	if (node_is_leaf(current_node))
	{
		for (int i = 0; i < 6; i++)
		{
			current_node->ropes[i] = ropes[i];
		}
	}
	else
	{
		optimize_ropes(current_node, ropes, nodes, bounds);

		int sl;
		int sr;
		if (current_node->split_axis == X_AXIS)
		{
			sl = LEFT_S;
			sr = RIGHT_S;
		}
		else if (current_node->split_axis == Y_AXIS)
		{
			sl = BOTTOM_S;
			sr = TOP_S;
		}
		else 
		{
			sl = BACK_S;
			sr = FRONT_S;
		}

		int left_ropes[6];
		int right_ropes[6];
		for (int i = 0; i < 6; i++)
		{
			left_ropes[i] = ropes[i];
			right_ropes[i] = ropes[i];
		}
		AABB left_bounds;
		AABB right_bounds;
		get_bounds(&left_bounds, &right_bounds, current_node->split_axis, bounds);
		left_ropes[sr] = current_node->right_index;
		build_ropes(&nodes[current_node->left_index], left_ropes, nodes, left_bounds);
		right_ropes[sl] = current_node->left_index;
		build_ropes(&nodes[current_node->right_index], right_ropes, nodes, right_bounds);
	}
}

KDTree kdtree_build(AABB* prims, int prim_amount, AABB bounds, int max_depth, int max_prims)
{
	ArrayList* nodes = arraylist_new(1);
	ArrayList* node_prims = arraylist_new(1);
	KDTreeNode* root = new_leaf(prim_amount);
	arraylist_add(nodes, root);
	int* root_prims = (int *) calloc(prim_amount, sizeof(int));
	for (int i = 0; i < prim_amount; i++)
	{
		root_prims[i] = i;
	}
	arraylist_add(node_prims, root_prims);
	build(root, 0, prims, prim_amount, max_depth, max_prims, nodes, node_prims, -1, bounds);

	KDTree tree;
	tree.node_amount = nodes->length;
	tree.nodes = (KDTreeNode *) calloc(tree.node_amount, sizeof(KDTreeNode));
	tree.total_prims = 0;
	for (int i = 0; i < nodes->length; i++)
	{
		KDTreeNode* node = (KDTreeNode *) arraylist_get(nodes, i);
		if (node_is_leaf(node))
		{
			node->prim_start = tree.total_prims;
			tree.total_prims += node->prim_amount;
		}
		tree.nodes[i] = *node;
		free(node);
	}
	arraylist_free(nodes);
	tree.node_prims = (int *) calloc(tree.total_prims, sizeof(int));
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
	int null_ropes[6] = {-1, -1, -1, -1, -1, -1};
	build_ropes(&tree.nodes[0], null_ropes, tree.nodes, bounds);
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