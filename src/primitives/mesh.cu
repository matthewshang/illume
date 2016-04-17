#include "mesh.h"

Mesh mesh_new()
{
	Mesh mesh;
	mesh.triangle_amount = 0;
	mesh.triangles = NULL;
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
		if (tokens[i]) free(tokens[i]);
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

	while(fgets(line, 100, file))
	{
		printf("line: %s\n", line);
		obj_get_token(line, &token);
		printf("token: %s\n", token);

		if (strcmp(token, TOKEN_VERTEX) == 0)
		{
			split_string(line, " ", VERTEX_COMPONENTS, tokens);
			for (int i = 1; i < VERTEX_COMPONENTS; i++)
			{
				printf("vertex: %s\n", tokens[i]);
			}
			split_string_finish(tokens, VERTEX_COMPONENTS);
		}
		else if (strcmp(token, TOKEN_FACE) == 0)
		{
			split_string(line, " ", FACE_COMPONENTS, tokens);
			for (int i = 1; i < FACE_COMPONENTS; i++)
			{
				printf("face: %s\n", tokens[i]);
			}
			split_string_finish(tokens, FACE_COMPONENTS);
		}

		if (token) free(token);
	}
	
	fclose(file);
	if (tokens) free(tokens);
}

void mesh_free(Mesh* mesh)
{
	if (mesh)
	{
		free(mesh->triangles);
	}
}