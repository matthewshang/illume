// Header files that include cuda code here to avoid C compiler issues

#include <curand.h>
#include <curand_kernel.h>

#include "kernel.h"

#include "vector3.h"
#include "ray.h"
#include "sphere.h"
#include "sample.h"
#include "scene.h"
#include "material.h"

__global__ 
void init_curand_states(curandState* states, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		curand_init(666420691337, index, 0, &states[index]);
	}
}

typedef struct
{
	float image_width;
	float camera_focus_plane;
	float camera_pixel_size;
	float camera_left;
	float camera_top;
} 
RenderInfo;

__global__
void init_rays(Ray* rays, int* ray_statuses, Vector3* ray_colors, RenderInfo* info, curandState* states, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		int x = index % (int) info->image_width;
		int y = index / (int) info->image_width;
		float left_edge = info->camera_left + info->camera_pixel_size * (float) x;
		float right_edge = left_edge + info->camera_pixel_size;
		float top_edge = info->camera_top - info->camera_pixel_size * (float) y;
		float bottom_edge = top_edge + info->camera_pixel_size;

		float r_x = left_edge + (right_edge - left_edge) * curand_uniform(&states[index]);
		float r_y = bottom_edge + (top_edge - bottom_edge) * curand_uniform(&states[index]);

		rays[index] = ray_create(vector3_create(0, 0, 0), vector3_create(r_x, r_y, info->camera_focus_plane));
		ray_statuses[index] = index;
		ray_colors[index] = vector3_create(1, 1, 1);
	}
}

__device__
static Intersection get_min_intersection(Scene* scene, Ray* ray)
{
	Intersection min = intersection_create_no_intersect();
	min.d = FLT_MAX;
	for (int i = 0; i < scene->sphere_amount; i++)
	{
		Intersection inter = sphere_ray_intersection(&scene->spheres[i], ray);

		if (inter.is_intersect == 1 && inter.d < min.d)
		{
			min = inter;
		}
	}
	return min;
}

__device__
static Vector3 get_background_color(Vector3 direction)
{
	float grad = (direction.x + 2) / 3;
	return vector3_create(grad, grad, grad);
	// return vector3_create(0.8, 0.8, 0.8);
}

__global__
void pathtrace_kernel(Vector3* final_colors, Ray* rays, int* ray_statuses, 
					  Vector3* ray_colors, Scene* scene, curandState* states, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ray_index = ray_statuses[index];
	if (index < N && ray_index != -1)
	{
		Intersection min = get_min_intersection(scene, &rays[ray_index]);
		if (min.is_intersect == 1)
		{
			if (vector3_length2(min.m.e) > 0)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.e);
				vector3_add_to(&final_colors[ray_index], ray_colors[ray_index]);
				ray_statuses[ray_index] = -1;
			}
			else
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.d);
				Vector3 new_origin = ray_position_along(rays[ray_index], min.d);
				Vector3 bias = vector3_mul(min.normal, 10e-4);
				vector3_add_to(&new_origin, bias);
				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 sample = sample_hemisphere_cosine(u1, u2);
				Vector3 new_direction = vector3_to_basis(sample, min.normal);
				ray_set(&rays[ray_index], new_origin, new_direction);
			}
		}
		else
		{
			Vector3 sky = get_background_color(rays[ray_index].d);
			vector3_mul_vector_to(&ray_colors[ray_index], sky);
			vector3_add_to(&final_colors[ray_index], ray_colors[ray_index]);
			ray_statuses[ray_index] = -1;
		}
	}
}

__global__
void set_bitmap(Vector3* final_colors, Pixel* pixels, float samples, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		pixels[index].red = (int) (255 * final_colors[index].x / samples);
		pixels[index].green = (int) (255 * final_colors[index].y / samples);
		pixels[index].blue = (int) (255 * final_colors[index].z / samples);
	}
}

static RenderInfo* allocate_render_info_gpu(int width, int height, float fov, float plane)
{
	RenderInfo i;
	i.image_width = width;
	float dim_ratio = (float) height / (float) width;
	float tan_half_fov = tanf(PI * fov / 360);
	i.camera_focus_plane = plane;	
	i.camera_pixel_size = tan_half_fov * 2 / (float) width;
	i.camera_left = -1 * plane * tan_half_fov;
	i.camera_top = dim_ratio * plane * tan_half_fov;
	RenderInfo *d_info;
	cudaMalloc(&d_info, sizeof(RenderInfo));
	cudaMemcpy(d_info, &i, sizeof(RenderInfo), cudaMemcpyHostToDevice);
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
	cudaMalloc(&d_final_colors, pixels_amount * sizeof(Vector3));
	cudaMemcpy(d_final_colors, h_final_colors, pixels_amount * sizeof(Vector3), cudaMemcpyHostToDevice);
	free(h_final_colors);
	return d_final_colors;
}

typedef struct
{
	Scene* d_scene;
	Sphere* d_spheres;
} 
SceneReference;

static SceneReference allocate_scene_gpu(Scene* scene)
{
	SceneReference ref;
	int spheres_size = sizeof(Sphere) * scene->sphere_amount;
	cudaMalloc(&ref.d_scene, sizeof(Scene));
	cudaMalloc(&ref.d_spheres, spheres_size);
	Sphere* h_spheres = scene->spheres;
	scene->spheres = ref.d_spheres;
	cudaMemcpy(ref.d_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);
	scene->spheres = h_spheres;
	cudaMemcpy(ref.d_spheres, scene->spheres, spheres_size, cudaMemcpyHostToDevice);
	return ref;
}

static void free_scene_gpu(SceneReference ref)
{
	cudaFree(ref.d_spheres);
	cudaFree(ref.d_scene);
}

void render_scene(Bitmap* bitmap, int samples)
{
	struct timespec tstart = {0, 0};
	struct timespec tend = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	Material white = material_diffuse(vector3_create(1, 1, 1));
	Material white_light = material_emissive(vector3_create(1, 1, 1));
	Material blue = material_diffuse(vector3_create(0, 0, 1));
	Material red = material_diffuse(vector3_create(1, 0, 0));

	Scene* scene = scene_new(4);
	scene->spheres[0] = sphere_create(10, vector3_create(0, -11, 8), white);
	scene->spheres[1] = sphere_create(1, vector3_create(0, 0, 8), white);
	scene->spheres[2] = sphere_create(0.5, vector3_create(-2, -0.75, 7), red);
	scene->spheres[3] = sphere_create(0.5, vector3_create(2, -0.75, 7), blue);
	// scene->spheres[4] = sphere_create(0.75, vector3_create(0, 4, 8), white_light);

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
	int pixels_amount = bitmap->width * bitmap->height;
	int threads_per_block = 256;
	int blocks_amount = (pixels_amount + threads_per_block - 1) / threads_per_block;

	curandState* d_states;
	cudaMalloc(&d_states, sizeof(curandState) * threads_per_block * blocks_amount);
	init_curand_states<<<blocks_amount, threads_per_block>>>(d_states, pixels_amount);

	RenderInfo* d_info = allocate_render_info_gpu(bitmap->width, bitmap->height, 70, 1);

	Vector3* d_final_colors = allocate_final_colors_gpu(pixels_amount);

	Vector3* d_ray_colors;
	cudaMalloc(&d_ray_colors, pixels_amount * sizeof(Vector3));

	int* d_ray_statuses;
	cudaMalloc(&d_ray_statuses, pixels_amount * sizeof(int));

	Ray* d_rays;
	cudaMalloc(&d_rays, sizeof(Ray) * pixels_amount);

	SceneReference ref = allocate_scene_gpu(scene);

	struct timespec tstart_render = {0, 0};
	struct timespec tend_render = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &tstart_render);

	for (int i = 0; i < samples; i++)
	{
		init_rays<<<blocks_amount, threads_per_block>>>
			(d_rays, d_ray_statuses, d_ray_colors, d_info, d_states, pixels_amount);

		for (int j = 0; j < 5; j++)
		{
			pathtrace_kernel<<<blocks_amount, threads_per_block>>>
				(d_final_colors, d_rays, d_ray_statuses, d_ray_colors, 
				 ref.d_scene, d_states, pixels_amount);		
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &tend_render);
	printf("Render Time: %f seconds\n", 
		    ((double) tend_render.tv_sec + 1.0e-9 * tend_render.tv_nsec) -
		    ((double) tstart_render.tv_sec + 1.0e-9 * tstart_render.tv_nsec));

	cudaFree(d_states);
	cudaFree(d_rays);
	cudaFree(d_info);
	cudaFree(d_ray_statuses);
	cudaFree(d_ray_colors);
	free_scene_gpu(ref);
	scene_free(scene);

	Pixel* h_pixels = bitmap->pixels;
	Pixel* d_pixels;
	cudaMalloc(&d_pixels, sizeof(Pixel) * pixels_amount);
	cudaMemcpy(d_pixels, h_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyHostToDevice);

	set_bitmap<<<blocks_amount, threads_per_block>>>(d_final_colors, d_pixels, (float) samples, pixels_amount);
	cudaMemcpy(h_pixels, d_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyDeviceToHost);

	cudaFree(d_final_colors);
	cudaFree(d_pixels);

	clock_gettime(CLOCK_MONOTONIC, &tend);
	printf("Total Time: %f seconds\n", 
		    ((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
		    ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));
}