#include "bvh.h"

#include "../math/vector3.h"
#include "../math/mathutils.h"

#include <stdio.h>
#include <time.h>

// Based on bvh described at http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

struct BVHNode
{
	AABB bounds;
	BVHNode* left;
	BVHNode* right;
	int* triangles;
	int tri_amount;
	bool is_leaf;
};

typedef struct
{
	AABB bounds;
	Vector3 center;
	int tri_index;
} BVHWork;

BVHWork bvhwork_create(AABB bounds, int tri_index)
{
	BVHWork work;
	work.bounds = bounds;
	work.center = vector3_mul(vector3_add(bounds.min, bounds.max), 0.5f);
	work.tri_index = tri_index;
	return work;
}

BVHNode* bvh_leaf_new(ArrayList* tris)
{
	BVHNode* leaf = (BVHNode*) calloc(1, sizeof(BVHNode));
	leaf->left = NULL;
	leaf->right = NULL;
	leaf->is_leaf = true;
	leaf->triangles = (int*) calloc(tris->length, sizeof(int));
	for (int i = 0; i < tris->length; i++)
	{
		leaf->triangles[i] = ((BVHWork*) arraylist_get(tris, i))->tri_index;
	}
	leaf->tri_amount = tris->length;
	return leaf;
}

BVHNode* bvh_inner_node_new(BVHNode* left, BVHNode* right)
{
	BVHNode* node = (BVHNode*) calloc(1, sizeof(BVHNode));
	node->left = left;
	node->right = right;
	node->triangles = NULL;
	node->tri_amount = 0;
	node->is_leaf = false;
	return node;
}

int report_counter = 0;

BVHNode* build(ArrayList* work, int depth, int tris_per_node, float pct)
{
	float pct_span = 11.f / powf(3.f, depth);

	if (work->length < tris_per_node || depth > BVH_STACK_SIZE)
	{
		return bvh_leaf_new(work);
	}
	
	AABB bounds = aabb_create();
	for (int i = 0; i < work->length; i++)
	{
		BVHWork* w = (BVHWork*) arraylist_get(work, i);
		aabb_update(&bounds, w->bounds.min);
		aabb_update(&bounds, w->bounds.max);
	}
	float xdim = bounds.max.x - bounds.min.x;
	float ydim = bounds.max.y - bounds.min.y;
	float zdim = bounds.max.z - bounds.min.z;
	
	float best_cost = work->length * ((xdim * ydim) + (ydim * zdim) + (zdim * xdim));
	float best_split = FLT_MAX;
	int best_axis = -1;

	for (int axis = 0; axis < 3; axis++)
	{
		float start;
		float end;
		float step;

		if (axis == 0)
		{
			start = bounds.min.x;
			end = bounds.max.x;
		}
		else if (axis == 1)
		{
			start = bounds.min.y;
			end = bounds.max.y;
		}
		else
		{
			start = bounds.min.z;
			end = bounds.max.z;
		}

		if (fabsf(end - start) < 1e-4)
		{
			continue;
		}

		step = (end - start) / (1024.f / ((float) depth + 1.0f));

		float pct_start = pct + axis * pct_span;
		float pct_step = pct_span / ((end - start - 2 * step) / step);

		for (float split = start + step; split < end - step; split += step)
		{
			if ((1023 & report_counter++) == 0)
			{
				printf("\b\b\b%02d%%", (int) pct_start); fflush(stdout);
			}
			pct_start += pct_step;

			AABB left_bounds = aabb_create();
			AABB right_bounds = aabb_create();
			int left_tri_amount = 0;
			int right_tri_amount = 0;
			
			for (int i = 0; i < work->length; i++)
			{
				BVHWork* w = (BVHWork*) arraylist_get(work, i);
				float center;
				if (axis == 0)
				{
					center = w->center.x;
				}
				else if (axis == 1)
				{
					center = w->center.y;
				}
				else
				{
					center = w->center.z;
				}

				if (center < split)
				{
					aabb_update(&left_bounds, w->bounds.min);
					aabb_update(&left_bounds, w->bounds.max);
					left_tri_amount++;
				}
				else
				{
					aabb_update(&right_bounds, w->bounds.min);
					aabb_update(&right_bounds, w->bounds.max);
					right_tri_amount++;
				}
			}

			if (left_tri_amount <= 1 || right_tri_amount <= 1)
			{
				continue;
			}

			float leftx = left_bounds.max.x - left_bounds.min.x;
			float lefty = left_bounds.max.y - left_bounds.min.y;
			float leftz = left_bounds.max.z - left_bounds.min.z;
			float leftsurf = leftx * lefty + lefty * leftz + leftz * leftx;

			float rightx = right_bounds.max.x - right_bounds.min.x;
			float righty = right_bounds.max.y - right_bounds.min.y;
			float rightz = right_bounds.max.z - right_bounds.min.z;
			float rightsurf = rightx * righty + righty * rightz + rightz * rightx;

			float cost = leftsurf * (float) left_tri_amount + rightsurf * (float) right_tri_amount;

			if (cost < best_cost)
			{
				best_cost = cost;
				best_split = split;
				best_axis = axis;
			}
		}
	}

	if (best_axis == -1)
	{
		return bvh_leaf_new(work);
	}

	ArrayList* left = arraylist_new(1);
	ArrayList* right = arraylist_new(1);
	AABB left_bounds = aabb_create();
	AABB right_bounds = aabb_create();

	for (int i = 0; i < work->length; i++)
	{
		BVHWork* w = (BVHWork*) arraylist_get(work, i);
		float center;
		if (best_axis == 0)
		{
			center = w->center.x;
		}
		else if (best_axis == 1)
		{
			center = w->center.y;
		}
		else
		{
			center = w->center.z;
		}

		if (center < best_split)
		{
			arraylist_add(left, w);
			aabb_update(&left_bounds, w->bounds.min);
			aabb_update(&left_bounds, w->bounds.max);
		}
		else
		{
			arraylist_add(right, w);
			aabb_update(&right_bounds, w->bounds.min);
			aabb_update(&right_bounds, w->bounds.max);
		}
	}

	if ((1023 & report_counter++) == 0)
	{
		printf("\b\b\b%2d%%", (int) (pct + 3.f * pct_span)); fflush(stdout);
	}
	BVHNode* left_node = build(left, depth + 1, tris_per_node, pct + 3.f * pct_span);

	if ((1023 & report_counter++) == 0)
	{
		printf("\b\b\b%2d%%", (int) (pct + 6.f * pct_span)); fflush(stdout);
	}
	BVHNode* right_node = build(right, depth + 1, tris_per_node, pct + 6.f * pct_span);

	arraylist_free(left);
	arraylist_free(right);
	left_node->bounds = left_bounds;
	right_node->bounds = right_bounds;
	return bvh_inner_node_new(left_node, right_node);
}

void node_free(BVHNode* node)
{
	if (node->is_leaf)
	{
		free(node->triangles);
		free(node);
	}
	else
	{
		node_free(node->left);
		node_free(node->right);
		free(node);
	}
}

void print_tree(BVHNode* node, int depth)
{
	if (node->is_leaf)
	{
		printf("leaf - depth %d - tris: ", depth);
		for (int i = 0; i < node->tri_amount; i++)
		{
			printf("%d, ", node->triangles[i]);
		}
		printf("\n");
	}
	else
	{
		printf("node - depth %d - bounds: \nmin: %f %f %f\nmax: %f %f %f\n", depth, node->bounds.min.x, node->bounds.min.y, node->bounds.min.z, node->bounds.max.x, node->bounds.max.y, node->bounds.max.z);
		print_tree(node->left, depth + 1);
		print_tree(node->right, depth + 1);
	}
}

int count_tris(BVHNode* node)
{
	if (node->is_leaf)
	{
		return node->tri_amount;
	}
	else
	{
		return count_tris(node->left) + count_tris(node->right);
	}
}

int count_nodes(BVHNode* node)
{
	if (node->is_leaf)
	{
		return 1;
	}
	else
	{
		return 1 + count_nodes(node->left) + count_nodes(node->right);
	}
}

void build_gpu_bvh(BVHNode* node, BVH* bvh, int* node_index, int* tri_index)
{
	int current_index = *node_index;
	bvh->nodes[current_index].bounds = node->bounds;
	if (node->is_leaf)
	{
		bvh->nodes[current_index].u.leaf.tri_amount = 0x80000000 | node->tri_amount;
		bvh->nodes[current_index].u.leaf.tri_start = *tri_index;
		for (int i = 0; i < node->tri_amount; i++)
		{
			bvh->tri_indices[(*tri_index)++] = node->triangles[i];
		}
	}
	else
	{
		int left_index = ++(*node_index);
		build_gpu_bvh(node->left, bvh, node_index, tri_index);
		int right_index = ++(*node_index);
		build_gpu_bvh(node->right, bvh, node_index, tri_index);
		bvh->nodes[current_index].u.node.left_node = left_index;
		bvh->nodes[current_index].u.node.right_node = right_index;
	}
}

void print_gpu_tree(BVH* bvh, int node_amount, int tri_amount)
{
	printf("\nPrinting gpu tree\n");
	printf("Tris - ");
	for (int i = 0; i < tri_amount; i++)
	{
		printf("%d, ", bvh->tri_indices[i]);
	}
	printf("\nNodes -\n");
	for (int i = 0; i < node_amount; i++)
	{
		if (bvh->nodes[i].u.leaf.tri_amount & 0x80000000) 
		{
			unsigned start = bvh->nodes[i].u.leaf.tri_start;
			unsigned size = bvh->nodes[i].u.leaf.tri_amount & 0x7fffffff;
			printf("%d  leaf - start: %u - size: %u - tris: ", i, start, size);
			for (unsigned j = start; j < start + size; j++)
			{
				printf("%d, ", bvh->tri_indices[j]);
			}
			printf("\n");
		}
		else
		{
			printf("%d  node - left: %u - right: %u\n", i, bvh->nodes[i].u.node.left_node, bvh->nodes[i].u.node.right_node);
			printf("   bounds: \n   min: %f %f %f\n   max: %f %f %f\n", bvh->nodes[i].bounds.min.x, bvh->nodes[i].bounds.min.y, bvh->nodes[i].bounds.min.z, bvh->nodes[i].bounds.max.x, bvh->nodes[i].bounds.max.y, bvh->nodes[i].bounds.max.z);
		}
	}
}

BVH build_bvh(std::vector<AABB> aabbs, AABB bounds, int tris_per_node)
{
	clock_t start = clock();
	clock_t diff;

	BVHWork* work = (BVHWork *) calloc(aabbs.size(), sizeof(BVHWork));
	ArrayList* work_list = arraylist_new(aabbs.size());
	for (int i = 0; i < aabbs.size(); i++)
	{
		work[i] = bvhwork_create(aabbs[i], i);
		arraylist_add(work_list, &work[i]);
	}
	printf("Building BVH...    "); fflush(stdout);
	BVHNode* root = build(work_list, 0, tris_per_node, 0.f);
	printf("\b\b\b100%%\n");
	root->bounds = bounds;
	free(work);
	arraylist_free(work_list);
	//print_tree(root, 0);
	BVH bvh;
	bvh.tri_index_amount = count_tris(root);
	bvh.tri_indices = (int*) calloc(bvh.tri_index_amount, sizeof(int));
	bvh.node_amount = count_nodes(root);
	bvh.nodes = (GPUNode*) calloc(bvh.node_amount, sizeof(GPUNode));
	int node_index = 0;
	int tri_index = 0;
	build_gpu_bvh(root, &bvh, &node_index, &tri_index);
	node_free(root);
	//print_gpu_tree(&bvh, bvh.node_amount, bvh.tri_index_amount);

	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Build time: %d.%d seconds\n", msec / 1000, msec % 1000);
	return bvh;
}

BVH bvh_create(std::vector<AABB> aabbs, AABB bounds, char* filename, int tris_per_node)
{
	FILE* fp = fopen(filename, "rb");
	if (!fp)
	{
		BVH bvh = build_bvh(aabbs, bounds, tris_per_node);
		fp = fopen(filename, "wb");
		if (!fp || (1 != fwrite(&bvh.node_amount, sizeof(int), 1, fp))
			|| (1 != fwrite(&bvh.tri_index_amount, sizeof(int), 1, fp))
			|| (bvh.node_amount != fwrite(bvh.nodes, sizeof(GPUNode), bvh.node_amount, fp))
			|| (bvh.tri_index_amount != fwrite(bvh.tri_indices, sizeof(int), bvh.tri_index_amount, fp)))
		{
			printf("Error: could not save bvh to %s\n", filename);
			return bvh;
		}
		printf("Saved prebuilt bvh to %s\n", filename);
		fclose(fp);
		return bvh;
	}
	else
	{
		BVH bvh;
		printf("Prebuilt bvh found, loading from %s\n", filename);
		if (1 != fread(&bvh.node_amount, sizeof(int), 1, fp) ||
			1 != fread(&bvh.tri_index_amount, sizeof(int), 1, fp))
		{
			printf("Error: could not read bvh\n");
			return build_bvh(aabbs, bounds, tris_per_node);
		}
		bvh.tri_indices = (int*) calloc(bvh.tri_index_amount, sizeof(int));
		bvh.nodes = (GPUNode*) calloc(bvh.node_amount, sizeof(GPUNode));
		if (bvh.node_amount != fread(bvh.nodes, sizeof(GPUNode), bvh.node_amount, fp) ||
			bvh.tri_index_amount != fread(bvh.tri_indices, sizeof(int), bvh.tri_index_amount, fp))
		{
			printf("Error: could not read bvh\n");
			return build_bvh(aabbs, bounds, tris_per_node);
		}
		fclose(fp);
		return bvh;
	}
}

void bvh_free(BVH* bvh)
{
	if (bvh->nodes) free(bvh->nodes);
	if (bvh->tri_indices) free(bvh->tri_indices);
}