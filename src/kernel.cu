#include <stdlib.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "kernel.h"
#include "bitmap.h"
#include "vector3.h"
#include "ray.h"
#include "sphere.h"

__global__ 
void init_curand_states(curandState* states, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		curand_init(666420691337, index, 0, &states[index]);
	}
}

__global__
void init_rays(Ray* rays, RenderInfo* info, curandState* states, int N)
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
	}
}

__global__
void pathtrace_kernel(Vector3* colors, Ray* rays, Sphere* sphere, curandState* states, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		if (sphere_ray_intersection(sphere, &rays[index]).is_intersect == 1)
		{
			Vector3 red = vector3_create(255, 0, 0);
			vector3_add_to(&colors[index], &red);
		}
		else
		{
			Vector3 blue = vector3_create(135, 206, 235);
			vector3_add_to(&colors[index], &blue);
		}
	}
}

__global__
void set_bitmap(Vector3* colors, Pixel* pixels, float samples, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		pixels[index].red = (int) (colors[index].x / samples);
		pixels[index].green = (int) (colors[index].y / samples);
		pixels[index].blue = (int) (colors[index].z / samples);
	}
}

static void init_render_info(RenderInfo* i, int width, int height, float fov, float plane)
{
	i->image_width = width;
	float dim_ratio = (float) height / (float) width;
	float tan_half_fov = tanf(PI * fov / 360);
	i->camera_focus_plane = plane;	
	i->camera_pixel_size = tan_half_fov * 2 / (float) width;
	i->camera_left = -1 * plane * tan_half_fov;
	i->camera_top = dim_ratio * plane * tan_half_fov;
}

void render_scene(Bitmap* bitmap, int samples)
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
	int N = bitmap->width * bitmap->height;
	int threads_per_block = 256;
	int blocks_amount = (N + threads_per_block - 1) / threads_per_block;

	RenderInfo info;
	init_render_info(&info, bitmap->width, bitmap->height, 90, 1);
	RenderInfo* d_info;
	cudaMalloc(&d_info, sizeof(RenderInfo));
	cudaMemcpy(d_info, &info, sizeof(RenderInfo), cudaMemcpyHostToDevice);

	// printf("RenderInfo initialized\n");

	curandState* d_states;
	cudaMalloc(&d_states, sizeof(curandState) * threads_per_block * blocks_amount);
	init_curand_states<<<blocks_amount, threads_per_block>>>(d_states, N);

	// printf("curand_states initialized\n");

	Sphere* sphere = sphere_new(1, vector3_create(0, 0, 5));
	Sphere* d_sphere;
	cudaMalloc(&d_sphere, sizeof(Sphere));
	cudaMemcpy(d_sphere, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);

	// printf("sphere initialized\n");

	Vector3* h_colors = (Vector3 *) malloc(sizeof(Vector3) * N);
	for (int i = 0; i < N; i++)
	{
		h_colors[i] = vector3_create(0, 0, 0);
	}

	// printf("h_colors done\n");
	Vector3* d_colors;
	cudaMalloc(&d_colors, N * sizeof(Vector3));

	// printf("d_colors malloc\n");
	cudaMemcpy(d_colors, h_colors, N * sizeof(Vector3), cudaMemcpyHostToDevice);

	// printf("colors initialized\n");

	Ray* d_rays;
	cudaMalloc(&d_rays, sizeof(Ray) * N);

	// printf("rays initialized\n");

	for (int i = 0; i < samples; i++)
	{
		init_rays<<<blocks_amount, threads_per_block>>>(d_rays, d_info, d_states, N);
		pathtrace_kernel<<<blocks_amount, threads_per_block>>>(d_colors, d_rays, d_sphere, d_states, N);		
	}

	// printf("tracing done\n");

	cudaFree(d_states);
	cudaFree(d_sphere);
	sphere_free(sphere);
	cudaFree(d_rays);
	cudaFree(d_info);


	Pixel* h_pixels = bitmap->pixels;
	Pixel* d_pixels;
	cudaMalloc(&d_pixels, sizeof(Pixel) * N);
	cudaMemcpy(d_pixels, h_pixels, sizeof(Pixel) * N, cudaMemcpyHostToDevice);

	// printf("pixels initialized\n");

	set_bitmap<<<blocks_amount, threads_per_block>>>(d_colors, d_pixels, (float) samples, N);
	cudaMemcpy(h_pixels, d_pixels, sizeof(Pixel) * N, cudaMemcpyDeviceToHost);

	// printf("bitmap done\n");

	cudaFree(d_colors);
	free(h_colors);
	cudaFree(d_pixels);
}