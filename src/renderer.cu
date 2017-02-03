#include "renderer.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "error_check.h"
#include "fresnel.h"
#include "material.h"
#include "microfacet.h"
#include "medium.h"
#include "primitives/sphere.h"
#include "primitives/mesh.h"
#include "primitives/mesh_instance.h"
#include "math/sample.h"
#include "math/vector3.h"
#include "math/ray.h"
#include "math/mathutils.h"
#include "math/matrix4.h"
#include "accel/bvh.h"
#include "jsonutils.h"

#include "intellisense.h"

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

Renderer::Renderer(rapidjson::Value& json, Scene* scene, int spp, int max_depth) : 
	m_scene(scene), m_spp(spp), m_max_depth(max_depth)
{
	auto renderer = json.FindMember("render_settings");
	if (renderer != json.MemberEnd())
	{
		JsonUtils::from_json(renderer->value, "ray_bias", m_ray_bias);

		auto res = renderer->value.FindMember("resolution");
		if (res != renderer->value.MemberEnd())
		{
			m_width = res->value.GetArray()[0].GetInt();
			m_height = res->value.GetArray()[1].GetInt();
		}
		else
		{
			printf("Renderer: resolution not found. Defaulting to (512, 512)\n");
			m_width = m_height = 512;
		}
	}
}

__global__ 
void init_curand_states(curandState* states, uint32_t hash, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		curand_init(hash + index, 0, 0, &states[index]);
	}
}

typedef struct
{
	float image_width;
	Camera camera;
	float camera_pixel_size;
	float camera_left;
	float camera_top;
} 
RenderInfo;

__global__
void init_rays(Ray* rays, int* ray_statuses, Vector3* ray_colors, Medium* ray_mediums, RenderInfo* info, curandState* states, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		RenderInfo i = *info;
		int x = index % (int) i.image_width;
		int y = index / (int) i.image_width;
		float left_edge = i.camera_left + i.camera_pixel_size * (float) x;
		float top_edge = i.camera_top - i.camera_pixel_size * (float) y;

		float r_x = left_edge + i.camera_pixel_size * curand_uniform(&states[index]);
		float r_y = top_edge - i.camera_pixel_size * curand_uniform(&states[index]);

		Vector3 pos = vector3_create(0, 0, 0);
		if (i.camera.aperture > FLT_EPSILON)
		{
			float u1 = curand_uniform(&states[index]);
			float u2 = curand_uniform(&states[index]);
			pos = vector3_mul(sample_circle(u1, u2), i.camera.aperture);
		}
		Vector3 origin = matrix4_mul_vector3(&i.camera.transform, pos, 1.f);
		Vector3 image_pos = matrix4_mul_vector3(&i.camera.transform, vector3_create(r_x, r_y, i.camera.dof), 1.f);
		rays[index] = ray_create(origin, vector3_sub(image_pos, origin));
		ray_statuses[index] = index;
		ray_colors[index] = vector3_create(1, 1, 1);
		ray_mediums[index] = medium_air();
	}
}

__device__
static void get_min_hit(Scene* scene, Ray ray, Hit* min)
{
	min->d = FLT_MAX;
	Hit inter;
	for (int i = 0; i < scene->sphere_amount; i++)
	{
		sphere_ray_intersect(scene->spheres[i], ray, &inter);

		if (inter.is_intersect == 1 && inter.d < min->d)
		{
			*min = inter;
		}
	}

	for (int i = 0; i < scene->instance_amount; i++)
	{
		int mesh_index = scene->instances[i].mesh_index;
		mesh_instance_ray_intersect(scene->instances + i, scene->meshes + mesh_index, ray, &inter);

		if (inter.is_intersect == 1 && inter.d < min->d)
		{
			*min = inter;
		}
	}
}

__global__
void pathtrace_kernel(Vector3* final_colors, Ray* rays, int* ray_statuses, Vector3* ray_colors, 
	Medium* ray_mediums, int depth, Scene* scene, curandState* states, float ray_bias, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ray_index = ray_statuses[index];

	if (index < N && ray_index != -1)
	{
		Scene local_scene = *scene;
		Hit min;
		hit_set_no_intersect(&min);
		get_min_hit(&local_scene, rays[ray_index], &min);

		if (ray_mediums[ray_index].active)
		{
			Medium medium = ray_mediums[ray_index];
			float u = curand_uniform(&states[ray_index]);
			float scatter_dist = -log(u) / medium.scattering;
			if (scatter_dist < min.d)
			{
				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 scatter_dir = vector3_to_basis(sample_henyey_greenstein(medium.g, u1, u2), rays[ray_index].d);
				ray_set(&rays[ray_index], ray_position_along(rays[ray_index], scatter_dist), scatter_dir);
				vector3_mul_vector_to(&ray_colors[ray_index], 
						vector3_create(expf(-1.0f * scatter_dist * medium.absorption.x),
									   expf(-1.0f * scatter_dist * medium.absorption.y),
									   expf(-1.0f * scatter_dist * medium.absorption.z)));
				if (vector3_length2(ray_colors[ray_index]) < 1e-4)
				{
					ray_statuses[index] = -1;
				}
				return;
			}
			else
			{
				vector3_mul_vector_to(&ray_colors[ray_index], 
						vector3_create(expf(-1.0f * min.d * medium.absorption.x),
									   expf(-1.0f * min.d * medium.absorption.y),
									   expf(-1.0f * min.d * medium.absorption.z)));

			}
		}

		if (min.is_intersect == 1)
		{
			Ray r = rays[ray_index];
			Vector3 new_dir;
			Vector3 norm_o = vector3_mul(min.normal, vector3_dot(min.normal, r.d) > 0 ? -1.0f : 1.0f);
			Vector3 new_origin = ray_position_along(r, min.d);

			if (min.m.type == MATERIAL_EMISSIVE)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
				vector3_add_to(&final_colors[ray_index], ray_colors[ray_index]);
				ray_statuses[index] = -1;
				new_dir = vector3_create(0, 0, 0);
			}
			else if (min.m.type == MATERIAL_DIFFUSE)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 sample = sample_hemisphere_cosine(u1, u2);
				new_dir = vector3_to_basis(sample, norm_o);
				vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
			}
			else if (min.m.type == MATERIAL_SPECULAR)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], vector3_create(0.99f, 0.99f, 0.99f));
				new_dir = vector3_reflect(r.d, norm_o);
				vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
			}
			else if (min.m.type == MATERIAL_REFRACTIVE)
			{
				float cosI = -vector3_dot(r.d, min.normal);
				float cosT = 0.0f;
				float F = Fresnel::dielectric(cosI, min.m.ior, cosT);

				if (F == 1.0f || curand_uniform(&states[ray_index]) < F)
				{
					new_dir = vector3_reflect(r.d, min.normal);
					vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
				}
				else
				{
					float eta = cosI < 0.0f ? min.m.ior : 1.0f / min.m.ior;
					ray_mediums[ray_index] = cosI > 0.0f ? min.m.medium : medium_air();
					new_dir = vector3_add(vector3_mul(r.d, eta),
						vector3_mul(norm_o, eta * fabsf(cosI) - cosT));
					vector3_add_to(&new_origin, vector3_mul(norm_o, -ray_bias));
				}

				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
			}
			else if (min.m.type == MATERIAL_COOKTORRANCE)
			{
				//float a = (1.2f - 0.2f * sqrtf(fabsf(vector3_dot(r.d, norm_o)))) * min.m.roughness;
				float a = min.m.roughness;
				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 m = vector3_to_basis(Microfacet::sample_Beckmann(a, u1, u2), norm_o);
				new_dir = vector3_reflect(r.d, m);

				Vector3 H = vector3_sub(new_dir, r.d);
				vector3_normalize(&H);

				float n1 = 1.0f;
				float n2 = min.m.ior;
				float nr = n1 / n2;
				float cosI = fabsf(vector3_dot(r.d, norm_o));
				float cosT = sqrtf(1.0f - nr * nr * (1.0f - cosI * cosI));
				float rperp = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
				rperp *= rperp;
				float rpar = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);
				rpar *= rpar;
				float F = (rperp + rpar) * 0.5f;

				float NdotL = fabsf(vector3_dot(norm_o, new_dir));
				float NdotV = cosI;
				float NdotH = fabsf(vector3_dot(norm_o, H));
				float VdotH = fabsf(vector3_dot(r.d, H));

				float G = fminf(1.0f, fminf(2.0f * NdotH * NdotV / VdotH, 2.0f * NdotH * NdotL / VdotH));

				float D = fminf(1.0f, fmaxf(0.0f, NdotH)) * expf((NdotH * NdotH - 1.0f) / (a * a * NdotH * NdotH)) / (ILLUME_PI * a * a * NdotH * NdotH * NdotH * NdotH);
				Vector3 reflectance = vector3_mul(min.m.c, fminf(1.0f, (F *  D * G) / (4.0f * NdotV)));
				//Vector3 reflectance = vector3_mul(min.m.c, (F *  D * G) / (4.0f * NdotV));

				vector3_mul_vector_to(&ray_colors[ray_index], reflectance);
				vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
			}
			else if (min.m.type == MATERIAL_ROUGHREFRACTIVE)
			{
				Vector3 wi = vector3_mul(r.d, -1.0f);
				float wiDotN = vector3_dot(wi, min.normal);
				float a = min.m.roughness * (1.2f - 0.2f * sqrtf(fabsf(wiDotN)));

				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 m = Microfacet::sample_Beckmann(a, u1, u2);
				m = vector3_to_basis(m, min.normal);

				float wiDotT = 0.0f;
				float wiDotM = vector3_dot(wi, m);
				float F = Fresnel::dielectric(wiDotM, min.m.ior, wiDotT);

				if (F == 1.0f || curand_uniform(&states[ray_index]) < F)
				{
					new_dir = vector3_reflect(r.d, m);
					if (wiDotN * vector3_dot(new_dir, min.normal) <= 0.0f)
					{
						ray_statuses[index] = -1;
						return;
					}
					vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
				}
				else
				{
					float eta = wiDotM < 0.0f ? min.m.ior : 1.0f / min.m.ior;
					ray_mediums[ray_index] = wiDotM > 0.0f ? min.m.medium : medium_air();
					new_dir = vector3_sub(
						vector3_mul(m, wiDotM * eta - (wiDotM > 0.0f ? 1.0f : -1.0f) * wiDotT),
						vector3_mul(wi, eta));
					if (wiDotN * vector3_dot(new_dir, min.normal) >= 0.0f)
					{
						ray_statuses[index] = -1;
						return;
					}
					vector3_add_to(&new_origin, vector3_mul(norm_o, -ray_bias));
				}

				float G = Microfacet::G_Beckmann(wi, new_dir, m, a);
				float weight = (G * fabsf(wiDotM)) / (fabsf(wiDotN) * fabsf(vector3_dot(m, min.normal)));
				vector3_mul_vector_to(&ray_colors[ray_index], vector3_mul(min.m.c, weight));
			}
			else if (min.m.type == MATERIAL_CONDUCTOR)
			{
				float cosI = -vector3_dot(r.d, min.normal);
				if (cosI <= 0)
				{
					ray_statuses[index] = -1;
					return;
				}
				new_dir = vector3_reflect(r.d, norm_o);
				vector3_add_to(&new_origin, vector3_mul(norm_o, ray_bias));
				vector3_mul_vector_to(&ray_colors[ray_index], Fresnel::conductor(min.m.c, min.m.k, cosI));
			}
			else if (min.m.type == MATERIAL_ROUGHCONDUCTOR)
			{
				Vector3 wi = vector3_mul(r.d, -1.0f);
				float wiDotN = vector3_dot(wi, min.normal);
				if (wiDotN <= 0)
				{
					ray_statuses[index] = -1;
					return;
				}
				float a = min.m.roughness * (1.2f - 0.2f * sqrtf(fabsf(wiDotN)));

				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 m = Microfacet::sample_Beckmann(a, u1, u2);
				m = vector3_to_basis(m, min.normal);

				Vector3 F = Fresnel::conductor(min.m.c, min.m.k, wiDotN);

				new_dir = vector3_reflect(r.d, m);
				if (wiDotN * vector3_dot(new_dir, min.normal) <= 0.0f)
				{
					ray_statuses[index] = -1;
					return;
				}
				float wiDotM = vector3_dot(wi, m);
				float G = Microfacet::G_Beckmann(wi, new_dir, m, a);
				float weight = (G * fabsf(wiDotM)) / (fabsf(wiDotN) * fabsf(vector3_dot(m, min.normal)));
				vector3_mul_vector_to(&ray_colors[ray_index], vector3_mul(F, weight));
				vector3_add_to(&new_origin, vector3_mul(m, ray_bias));
			}

			ray_set(&rays[ray_index], new_origin, new_dir);
		}
		else
		{
			vector3_mul_vector_to(&ray_colors[ray_index], scene->sky_color);
			vector3_add_to(&final_colors[ray_index], ray_colors[ray_index]);
			ray_statuses[index] = -1;
		}


	}
}

static void compact_pixels(int* d_ray_statuses, int* h_ray_statuses, int* active_pixels)
{
	int pixels = *active_pixels;
	int size = pixels * sizeof(int); 
	HANDLE_ERROR( cudaMemcpy(h_ray_statuses, d_ray_statuses, size, cudaMemcpyDeviceToHost) );

	int left = 0;
	int right = pixels - 1;
	while (left < right)
	{
		while (h_ray_statuses[left] != -1 && left < pixels)
		{
			left++;
		}
		while (h_ray_statuses[right] == -1 && right >= 0)
		{
			right--;
		}

		if (left < right)
		{
			h_ray_statuses[left] = h_ray_statuses[right];
			h_ray_statuses[right] = -1;
		}
	}

	left = 0;
	while (left < pixels && h_ray_statuses[left] != -1)
	{
		left++;
	}
	*active_pixels = left;

	HANDLE_ERROR( cudaMemcpy(d_ray_statuses, h_ray_statuses, size, cudaMemcpyHostToDevice) );
}

__global__
void set_bitmap(Vector3* final_colors, Pixel* pixels, float samples, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		float gamma = 1 / 2.2;
		Vector3 corrected = vector3_mul(final_colors[index], 1 / samples);
		corrected = vector3_max(vector3_min(corrected, 1), 0);
		corrected = vector3_pow(corrected, gamma);
		pixels[index].red = (int) (255 * corrected.x);
		pixels[index].green = (int) (255 * corrected.y);
		pixels[index].blue = (int) (255 * corrected.z);
	}
}

static RenderInfo* allocate_render_info_gpu(int width, int height, Camera camera)
{
	RenderInfo i;
	i.image_width = width;
	float dim_ratio = (float) height / (float) width;
	float tan_half_fov = tanf(ILLUME_PI * camera.fov / 360);
	i.camera = camera;
	float dofmfov = camera.dof * tan_half_fov;
	i.camera_pixel_size = dofmfov * 2 / (float) width;
	i.camera_left = -1 * dofmfov;
	i.camera_top = dim_ratio * dofmfov;
	RenderInfo *d_info;
	HANDLE_ERROR( cudaMalloc(&d_info, sizeof(RenderInfo)) );
	HANDLE_ERROR( cudaMemcpy(d_info, &i, sizeof(RenderInfo), cudaMemcpyHostToDevice) );
	return d_info;
}

static Vector3* allocate_final_colors_gpu(int pixels_amount)
{
	Vector3* h_final_colors = (Vector3 *) malloc(sizeof(Vector3) * pixels_amount);
	for (int i = 0; i < pixels_amount; i++)
	{
		h_final_colors[i] = vector3_create(0, 0, 0);
	}
	Vector3* d_final_colors;
	HANDLE_ERROR( cudaMalloc(&d_final_colors, pixels_amount * sizeof(Vector3)) );
	HANDLE_ERROR( cudaMemcpy(d_final_colors, h_final_colors, pixels_amount * sizeof(Vector3), cudaMemcpyHostToDevice) );
	free(h_final_colors);
	return d_final_colors;
}

typedef struct
{
	Scene* d_scene;
	Sphere* d_spheres;
	Mesh* d_meshes;
	int mesh_amount;
	Triangle** d_triangle_pointers;
	MeshInstance* d_instances;
	GPUNode** d_bvh_nodes;
	int** d_bvh_indices;
} 
SceneReference;

static SceneReference allocate_scene_gpu(Scene* scene)
{
	SceneReference ref;
	int spheres_size = sizeof(Sphere) * scene->sphere_amount;
	int meshes_size = sizeof(Mesh) * scene->mesh_amount;
	int instances_size = sizeof(MeshInstance) * scene->instance_amount;
	ref.mesh_amount = scene->mesh_amount;

	HANDLE_ERROR( cudaMalloc(&ref.d_scene, sizeof(Scene)) );
	HANDLE_ERROR( cudaMalloc(&ref.d_spheres, spheres_size) );
	HANDLE_ERROR( cudaMalloc(&ref.d_instances, instances_size) );
	HANDLE_ERROR( cudaMalloc(&ref.d_meshes, meshes_size) );
	ref.d_triangle_pointers = (Triangle **) calloc(scene->mesh_amount, sizeof(Triangle *));
	Triangle** h_triangle_pointers = (Triangle **) calloc(scene->mesh_amount, sizeof(Triangle *));

	ref.d_bvh_nodes = (GPUNode **) calloc(scene->mesh_amount, sizeof(GPUNode*));
	GPUNode** h_bvh_nodes = (GPUNode **) calloc(scene->mesh_amount, sizeof(GPUNode*));
	ref.d_bvh_indices = (int **) calloc(scene->mesh_amount, sizeof(int*));
	int** h_bvh_indices = (int **) calloc(scene->mesh_amount, sizeof(int*));

	for (int i = 0; i < scene->mesh_amount; i++)
	{
		Mesh* mesh_p = &scene->meshes[i];
		Mesh mesh = *mesh_p;

		int triangles_size = mesh.triangle_amount * sizeof(Triangle);
		HANDLE_ERROR( cudaMalloc(&ref.d_triangle_pointers[i], triangles_size) );
		HANDLE_ERROR( cudaMemcpy(ref.d_triangle_pointers[i], mesh.triangles, triangles_size, cudaMemcpyHostToDevice) );
		h_triangle_pointers[i] = mesh.triangles;
		mesh_p->triangles = ref.d_triangle_pointers[i];	

		int nodes_size = mesh.bvh.node_amount * sizeof(GPUNode);
		HANDLE_ERROR(cudaMalloc(&ref.d_bvh_nodes[i], nodes_size));
		HANDLE_ERROR(cudaMemcpy(ref.d_bvh_nodes[i], mesh.bvh.nodes, nodes_size, cudaMemcpyHostToDevice));
		h_bvh_nodes[i] = mesh.bvh.nodes;
		mesh_p->bvh.nodes = ref.d_bvh_nodes[i];

		int indices_size = mesh.bvh.tri_index_amount * sizeof(int);
		HANDLE_ERROR(cudaMalloc(&ref.d_bvh_indices[i], indices_size));
		HANDLE_ERROR(cudaMemcpy(ref.d_bvh_indices[i], mesh.bvh.tri_indices, indices_size, cudaMemcpyHostToDevice));
		h_bvh_indices[i] = mesh.bvh.tri_indices;
		mesh_p->bvh.tri_indices = ref.d_bvh_indices[i];
	}

	Sphere* h_spheres = scene->spheres;
	Mesh* h_meshes = scene->meshes;
	MeshInstance* h_instances = scene->instances;
	scene->spheres = ref.d_spheres;
	scene->meshes = ref.d_meshes;
	scene->instances = ref.d_instances;
	HANDLE_ERROR( cudaMemcpy(ref.d_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice) );
	scene->spheres = h_spheres;
	scene->meshes = h_meshes;
	scene->instances = h_instances;
	HANDLE_ERROR( cudaMemcpy(ref.d_spheres, scene->spheres, spheres_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(ref.d_instances, scene->instances, instances_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(ref.d_meshes, scene->meshes, meshes_size, cudaMemcpyHostToDevice) );
	for (int i = 0; i < scene->mesh_amount; i++)
	{
		scene->meshes[i].triangles = h_triangle_pointers[i];
		scene->meshes[i].bvh.nodes = h_bvh_nodes[i];
		scene->meshes[i].bvh.tri_indices = h_bvh_indices[i];
	}
	free(h_triangle_pointers);
	free(h_bvh_nodes);
	free(h_bvh_indices);
	return ref;
}

static void free_scene_gpu(SceneReference ref)
{
	HANDLE_ERROR( cudaFree(ref.d_spheres) );
	HANDLE_ERROR( cudaFree(ref.d_instances) );
	HANDLE_ERROR( cudaFree(ref.d_meshes) );
	HANDLE_ERROR( cudaFree(ref.d_scene) );
	for (int i = 0; i < ref.mesh_amount; i++)
	{
		HANDLE_ERROR( cudaFree(ref.d_triangle_pointers[i]) );
		HANDLE_ERROR( cudaFree(ref.d_bvh_nodes[i]) );
		HANDLE_ERROR( cudaFree(ref.d_bvh_indices[i]) );
	}
	free(ref.d_triangle_pointers);
	free(ref.d_bvh_nodes);
	free(ref.d_bvh_indices);
}

static void start_timer(cudaEvent_t* start, cudaEvent_t* stop)
{
	HANDLE_ERROR( cudaEventCreate(start) );
	HANDLE_ERROR( cudaEventCreate(stop) );
	HANDLE_ERROR( cudaEventRecord(*start, 0) );
}

static void end_timer(cudaEvent_t* start, cudaEvent_t* stop, float* time)
{
	HANDLE_ERROR( cudaEventRecord(*stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(*stop) );
	HANDLE_ERROR( cudaEventElapsedTime(time, *start, *stop) );
}

uint32_t wang_hash(uint32_t a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void Renderer::render_to_bitmap(Bitmap* bitmap)
{
	cudaEvent_t render_start;
	cudaEvent_t render_stop;
	start_timer(&render_start, &render_stop);

	HANDLE_ERROR( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024) );
	int pixels_amount = m_width * m_height;
	int threads_per_block = 256;
	int blocks_amount = (pixels_amount + threads_per_block - 1) / threads_per_block;

	curandState* d_states;
	HANDLE_ERROR( cudaMalloc(&d_states, sizeof(curandState) * threads_per_block * blocks_amount) );

	RenderInfo* d_info = 
		allocate_render_info_gpu(m_width, m_height, m_scene->camera);

	Vector3* d_final_colors = allocate_final_colors_gpu(pixels_amount);

	Vector3* d_ray_colors;
	HANDLE_ERROR( cudaMalloc(&d_ray_colors, pixels_amount * sizeof(Vector3)) );

	int* d_ray_statuses;
	HANDLE_ERROR( cudaMalloc(&d_ray_statuses, pixels_amount * sizeof(int)) );

	Medium* d_ray_mediums;
	HANDLE_ERROR( cudaMalloc(&d_ray_mediums, pixels_amount * sizeof(Medium)) );

	Ray* d_rays;
	HANDLE_ERROR( cudaMalloc(&d_rays, sizeof(Ray) * pixels_amount) );

	SceneReference ref = allocate_scene_gpu(m_scene);

	int* h_ray_statuses = (int *) calloc(pixels_amount, sizeof(int));

	printf("Rendering...    "); fflush(stdout);
	int last_progress = -1;
	float progress_step = 100.0f / (float) m_spp;
	cudaEvent_t start, stop;
	for (int i = 0; i < m_spp; i++)
	{
		start_timer(&start, &stop);
		init_curand_states KERNEL_ARGS2(blocks_amount, threads_per_block) (d_states, wang_hash(i), pixels_amount);

		init_rays KERNEL_ARGS2(blocks_amount, threads_per_block)
			(d_rays, d_ray_statuses, d_ray_colors, d_ray_mediums, d_info, d_states, pixels_amount);

		int active_pixels = pixels_amount;
		int blocks = blocks_amount;

		for (int j = 0; j < m_max_depth; j++)
		{
			pathtrace_kernel KERNEL_ARGS2(blocks, threads_per_block)
				(d_final_colors, d_rays, d_ray_statuses, d_ray_colors, d_ray_mediums,
				 j, ref.d_scene, d_states, m_ray_bias, active_pixels);		
			compact_pixels(d_ray_statuses, h_ray_statuses, &active_pixels);
			blocks = (active_pixels + threads_per_block - 1) / threads_per_block;
		}
		int progress = (int) ((float) i * progress_step);
		if (progress != last_progress)
		{
			printf("\b\b\b%02d%%", progress); fflush(stdout);
			last_progress = progress;
		}
		
	}
	printf("\b\b\b100%%\n");

	HANDLE_ERROR( cudaFree(d_states) );
	HANDLE_ERROR( cudaFree(d_rays) );
	HANDLE_ERROR( cudaFree(d_info) );
	HANDLE_ERROR( cudaFree(d_ray_statuses) );
	HANDLE_ERROR( cudaFree(d_ray_colors) );
	HANDLE_ERROR( cudaFree(d_ray_mediums) );
	free_scene_gpu(ref);
	free(h_ray_statuses);

	Pixel* h_pixels = bitmap->pixels;
	Pixel* d_pixels;
	HANDLE_ERROR( cudaMalloc(&d_pixels, sizeof(Pixel) * pixels_amount) );
	HANDLE_ERROR( cudaMemcpy(d_pixels, h_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyHostToDevice) );

	set_bitmap KERNEL_ARGS2(blocks_amount, threads_per_block) 
		(d_final_colors, d_pixels, (float) m_spp, pixels_amount);
	HANDLE_ERROR( cudaMemcpy(h_pixels, d_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyDeviceToHost) );

	HANDLE_ERROR( cudaFree(d_final_colors) );
	HANDLE_ERROR( cudaFree(d_pixels) );

	float render_time;
	end_timer(&render_start, &render_stop, &render_time);

	printf("Render time: %f seconds\n", 1e-3 * (double) render_time);
}

int Renderer::get_width()
{
	return m_width;
}

int Renderer::get_height()
{
	return m_height;
}