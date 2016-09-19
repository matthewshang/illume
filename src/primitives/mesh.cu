  #include "mesh.h"

typedef struct
{
	ArrayList* vertices;
	ArrayList* triangles;
	ArrayList* aabbs;
}
tmp_mesh;

void mesh_free(Mesh* mesh)
{
	if (mesh)
	{
		if (mesh->triangles)
		{
			free(mesh->triangles);
		}
		kdtree_free(&mesh->tree);
		free(mesh);
	}
}

static void obj_get_token(char* line, char** ret)
{
	if (!line) 
	{
		return;
	}
	char* dup = strdup(line);
	char* p = strtok(dup, " ");
	if (p)
	{
		*ret = strdup(p);
	}
	free(dup);
}

static void split_string(const char* string, const char* delim, int tokens, char** ret)
{
	char* dup = strdup(string);
	char* p = strtok(dup, delim);
	int i = 0;
	while (i < tokens && p != NULL)
	{
		ret[i++] = strdup(p);
		p = strtok(NULL, delim);
	}

	free(dup);
}

static void split_string_finish(char** tokens, int amount)
{
	for (int i = 0; i < amount; i++)
	{
		if (tokens[i]) 
		{	
			free(tokens[i]);
		}
	}
}

static void fix_aabb(AABB* aabb)
{
	if (aabb->max.x - aabb->min.x < FLT_EPSILON)
	{
		aabb->max.x += FLT_EPSILON;
	}
	if (aabb->max.y - aabb->min.y < FLT_EPSILON)
	{
		aabb->max.y += FLT_EPSILON;
	}
	if (aabb->max.z - aabb->min.z < FLT_EPSILON)
	{
		aabb->max.z += FLT_EPSILON;
	}
}

static void load_obj(Mesh* mesh, const char* path, tmp_mesh* tmp)
{
	FILE* file;
	file = fopen(path, "rt");
	if (!file)
	{
		printf("mesh_load_obj: cannot read file %s\n", path);
		return;
	}

	char line[100];
	char* token;
	char** tokens = (char **) calloc(OBJ_TOKENS, sizeof(char *));
	tmp->vertices = arraylist_new(3);
	tmp->triangles = arraylist_new(1);
	tmp->aabbs = arraylist_new(1);

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			arraylist_add(tmp->vertices, vector3_new(strtof(tokens[1], NULL),
												strtof(tokens[2], NULL),
											    strtof(tokens[3], NULL)));
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
		else if (strcmp(token, TOKEN_FACE) == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
			int i0 = strtol(tokens[1], NULL, 10) - 1;
			int i1 = strtol(tokens[2], NULL, 10) - 1;
			int i2 = strtol(tokens[3], NULL, 10) - 1;
			Vector3 v0 = *((Vector3 *) arraylist_get(tmp->vertices, i0));
			Vector3 v1 = *((Vector3 *) arraylist_get(tmp->vertices, i1));
			Vector3 v2 = *((Vector3 *) arraylist_get(tmp->vertices, i2));
			AABB* aabb = (AABB *) calloc(1, sizeof(AABB));
			*aabb = aabb_create();
			aabb_update(aabb, v0);
			aabb_update(aabb, v1);
			aabb_update(aabb, v2);
			arraylist_add(tmp->aabbs, aabb);
			arraylist_add(tmp->triangles, triangle_new(v0, v1, v2));
			split_string_finish(tokens, FACE_COMPONENTS);
		}

		if (token) 
		{
			free(token);
		}
	}
	if (tokens)
	{
		free(tokens);
	}

	fclose(file);
}

static void copy_triangles(tmp_mesh* tmp, Mesh* mesh)
{
	mesh->triangles = (Triangle *) calloc(tmp->triangles->length, sizeof(Triangle));
	if (mesh->triangles)
	{
		for (int i = 0; i < tmp->triangles->length; i++)
		{
			Triangle* triangle = (Triangle *) arraylist_get(tmp->triangles, i);
			mesh->triangles[i] = *triangle;
			triangle_free(triangle);
		}
		mesh->triangle_amount = tmp->triangles->length;
		arraylist_free(tmp->triangles);
	}
	else
	{
		printf("mesh_load_obj: allocation of mesh tris failed");
	}

}

static void build_mesh_bounds(tmp_mesh* tmp, Mesh* mesh)
{
	mesh->aabb = aabb_create();
	for (int i = 0; i < tmp->vertices->length; i++)
	{
		Vector3* v = (Vector3 *) arraylist_get(tmp->vertices, i);
		aabb_update(&mesh->aabb, *v);
		vector3_free(v);
	}
	arraylist_free(tmp->vertices);
}

static void build_tree(tmp_mesh* tmp, Mesh* mesh, int max_depth, int max_per_node)
{
	AABB* final_aabbs = (AABB *) calloc(tmp->aabbs->length, sizeof(AABB));
	for (int i = 0; i < tmp->aabbs->length; i++)
	{
		AABB* current = (AABB *) arraylist_get(tmp->aabbs, i);
		fix_aabb(current);
		final_aabbs[i] = *current;
	}
	mesh->tree = kdtree_build(final_aabbs, tmp->aabbs->length, mesh->aabb, max_depth, max_per_node);
	free(final_aabbs);
	KDTree tree = mesh->tree;
	for (int i = 0; i < tree.node_amount; i++)
	{
		printf("%d :: %d %d\n", i, tree.nodes[i].left_index, tree.nodes[i].right_index);
		printf("%d %d %d %f\n", tree.nodes[i].prim_start, tree.nodes[i].prim_amount, tree.nodes[i].split_axis, tree.nodes[i].split_value);
		for (int j = 0; j < tree.nodes[i].prim_amount; j++)
		{
			printf("%d ", tree.node_prims[tree.nodes[i].prim_start + j]);
		}
		printf("\n");
		for (int j = 0; j < 6; j++)	printf("%d ", tree.nodes[i].ropes[j]);
		printf("\n\n");
	}
}

Mesh* mesh_new(const char* path, int tree_max_depth, int tree_max_per_node)
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;
	mesh->is_tree_built = 0;
	tmp_mesh tmp;
	load_obj(mesh, path, &tmp);
	copy_triangles(&tmp, mesh);
	build_mesh_bounds(&tmp, mesh);
	if (tree_max_depth > 0)
	{
		build_tree(&tmp, mesh, tree_max_depth, tree_max_per_node);
		printf("mesh_new: built tree for mesh %s\n", path);
		mesh->is_tree_built = 1;
	}
	for (int i = 0; i < tmp.aabbs->length; i++)
	{
		free((AABB *) arraylist_get(tmp.aabbs, i));
	}
	arraylist_free(tmp.aabbs);
	return mesh;
}

__device__
static int point_in_triangle(float bx, float by, float cx, float cy, float px, float py, float area)
{
	float ABP = tri_area_times_two(0, 0, bx, by, px, py);
	float BCP = tri_area_times_two(bx, by, cx, cy, px, py);
	float CAP = tri_area_times_two(cx, cy, 0, 0, px, py);
	return (ABP + BCP + CAP < area);
}

__device__
static float triangle_ray_intersect(Triangle tri, Ray ray)
{
	float d = vector3_dot(tri.n, vector3_sub(tri.v0, ray.o)) / vector3_dot(tri.n, ray.d);
	if (d < 0)
	{
		return -FLT_MAX;
	}
	Vector3 point = ray_position_along(ray, d);
	Vector3 p0 = vector3_sub(point, tri.v0);

	// one point of triangle is at origin 
	if (point_in_triangle(tri.t1x, tri.t1y, tri.t2x, tri.t2y,
						  vector3_dot(p0, tri.ex), vector3_dot(p0, tri.ey), tri.area))
	{
		return d;
	}

	return -FLT_MAX;
}

__device__
static int left_of_split(Vector3 point, int axis, float value)
{
	float v[] = { point.x, point.y, point.z };
	return v[axis] < value;
}

__device__
static int get_exit_side(AABB aabb, Vector3 exit_point, int* ropes)
{
	if (fabsf(aabb.min.x - exit_point.x) <= 2 * FLT_EPSILON)
	{
		return ropes[LEFT_S];
	}
	else if (fabsf(aabb.max.x - exit_point.x) <= 2 * FLT_EPSILON)
	{
		return ropes[RIGHT_S];
	}
	else if (fabsf(aabb.min.y - exit_point.y) <= 2 * FLT_EPSILON)
	{
		return ropes[BOTTOM_S];
	}
	else if (fabsf(aabb.max.y - exit_point.y) <= 2 * FLT_EPSILON)
	{
		return ropes[TOP_S];
	}
	else if (fabsf(aabb.min.z - exit_point.z) <= 2 * FLT_EPSILON)
	{
		return ropes[BACK_S];
	}
	else if (fabsf(aabb.max.z - exit_point.z) <= 2 * FLT_EPSILON)
	{
		return ropes[FRONT_S];
	}
	return -1;
}

__device__
static Hit kdtree_ray_intersect(Mesh* mesh, Ray ray)
{
	KDTree tree = mesh->tree;
	float entry = -FLT_MAX;
	float exit = FLT_MAX;
	KDTreeNode* node = &tree.nodes[0];
	aabb_ray_get_points(node->aabb, ray, &entry, &exit);

	//if (entry > 0)
	//{
	//	Vector3 point = ray_position_along(ray, entry);
	//	printf("min: %f %f %f max: %f %f %f entry: %f exit: %f point: %f %f %f\n", node->aabb.min.x, node->aabb.min.y, node->aabb.min.z, node->aabb.max.x, node->aabb.max.y, node->aabb.max.z, entry, exit, point.x, point.y, point.z);
	//}
	Hit min = hit_create_no_intersect();
	min.d = FLT_MAX;

	float last_entry = -FLT_MAX;
	int iteration = 0;
	while (entry < exit && entry > last_entry)
	{
		last_entry = entry;
		//printf("min: %f %f %f max: %f %f %f entry: %f exit: %f\n", node->aabb.min.x, node->aabb.min.y, node->aabb.min.z, node->aabb.max.x, node->aabb.max.y, node->aabb.max.z, entry, exit);

		Vector3 point = ray_position_along(ray, entry);
		while (!kdtree_node_is_leaf(node))
		{
			if (left_of_split(point, node->split_axis, node->split_value))
			{
				node = &tree.nodes[node->left_index];
			}
			else
			{
				node = &tree.nodes[node->right_index];
			}
		}
		//if (node->aabb.min.x < ray.o.x && node->aabb.min.y < ray.o.y && node->aabb.min.z < ray.o.z && node->aabb.max.x > ray.o.x && node->aabb.max.y > ray.o.y && node->aabb.max.z > ray.o.z)
		//{
		//	printf("inside %f\n", aabb_ray_exit(node->aabb, ray));
		//}
		//if (aabb_ray_exit(node->aabb, ray) == -FLT_MAX)
		{
			//if (node->aabb.min.x < ray.o.x && node->aabb.min.y < ray.o.y && node->aabb.min.z < ray.o.z && node->aabb.max.x > ray.o.x && node->aabb.max.y > ray.o.y && node->aabb.max.z > ray.o.z)
			{
				//printf("min: %f %f %f max: %f %f %f\n origin %f %f %f\n", node->aabb.min.x, node->aabb.min.y, node->aabb.min.z, node->aabb.max.x, node->aabb.max.y, node->aabb.max.z, ray.o.x, ray.o.y, ray.o.z);

				//printf("inside %f %f %f\n", ray.o.x, ray.o.y, ray.o.z);
			}
		}

		for (int i = 0; i < node->prim_amount; i++)
		{
			int prim_index = tree.node_prims[node->prim_start + i];
			Triangle tri = mesh->triangles[prim_index];
			float d = triangle_ray_intersect(tri, ray);
			if (d > 0 && d < exit && d > entry)
			{
				min.d = d;
				min.normal = tri.n;
				min.is_intersect = 1;
				exit = d;
			}
		}

		entry = aabb_ray_exit(node->aabb, ray);

		int rope_index = get_exit_side(node->aabb, ray_position_along(ray, entry), node->ropes);
		//Vector3 point2 = ray_position_along(ray, entry);
		//printf("min: %f %f %f max: %f %f %f point: %f %f %f rope: %d\n", node->aabb.min.x, node->aabb.min.y, node->aabb.min.z, node->aabb.max.x, node->aabb.max.y, node->aabb.max.z, point2.x, point2.y, point2.z, rope_index);

		if (rope_index == -1)
		{
			break;
		}

		node = &tree.nodes[rope_index];
		//if (iteration == 1 && min.is_intersect == 1)
		//	printf("%d %f %f %f %d\n", iteration, last_entry, entry, exit, rope_index);
		iteration++;
	}
		//printf("%f %f %f - %f %f %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z);
		//printf("min - d: %f normal: %f %f %f is: %d\n", min.d, min.normal.x, min.normal.y, min.normal.z, min.is_intersect);
	return min;
}

__device__
Hit mesh_ray_intersect(Mesh* mesh, Ray ray)
{
	if (!mesh->is_tree_built)
	{
		Hit min = hit_create_no_intersect();
		min.d = FLT_MAX;
		for (int i = 0; i < mesh->triangle_amount; i++)
		{
			Triangle tri = mesh->triangles[i];
			float inter_d = triangle_ray_intersect(tri, ray);

			if (inter_d > 0 && inter_d < min.d)
			{
				min.d = inter_d;
				min.normal = tri.n;
				min.is_intersect = 1;
			}
		}

		return min;
	}
	
	//return kdtree_ray_intersect(mesh, ray);
}