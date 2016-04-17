#include "mesh.h"

Mesh* mesh_new()
{
	Mesh* mesh = (Mesh *) calloc(1, sizeof(Mesh));
	if (!mesh)
	{
		return NULL;
	}
	mesh->triangle_amount = 0;
	mesh->triangles = NULL;
	return mesh;
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

void mesh_load_obj(Mesh* mesh, const char* path)
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
	ArrayList* vertices = arraylist_new(3);
	ArrayList* triangles = arraylist_new(1);

	while(fgets(line, 100, file))
	{
		obj_get_token(line, &token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			arraylist_add(vertices, vector3_new(strtof(tokens[1], NULL),
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
			Vector3* v0 = (Vector3 *) arraylist_get(vertices, i0);
			Vector3* v1 = (Vector3 *) arraylist_get(vertices, i1);
			Vector3* v2 = (Vector3 *) arraylist_get(vertices, i2);
			arraylist_add(triangles, triangle_new(*v0, *v1, *v2));
			split_string_finish(tokens, FACE_COMPONENTS);
		}

		if (token) 
		{
			free(token);
		}
	}
	mesh->triangles = (Triangle *) calloc(triangles->length, sizeof(Triangle));
	if (mesh->triangles)
	{
		for (int i = 0; i < triangles->length; i++)
		{
			mesh->triangles[i] = *((Triangle *) arraylist_get(triangles, i));
		}
		mesh->triangle_amount = triangles->length;
	}
	else
	{
		printf("mesh_load_obj: allocation of mesh tris failed");
	}

	for (int i = 0; i < vertices->length; i++)
	{
		vector3_free((Vector3 *) arraylist_get(vertices, i));
	}
	arraylist_free(vertices);
	for (int i = 0; i < triangles->length; i++)
	{
		triangle_free((Triangle *) arraylist_get(triangles, i));
	}
	arraylist_free(triangles);
	fclose(file);
	if (tokens)
	{ 
		free(tokens);
	}
}

void mesh_free(Mesh* mesh)
{
	if (mesh)
	{
		free(mesh->triangles);
	}
}