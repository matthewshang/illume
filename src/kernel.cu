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
		int y = index / (int) info->image_height;
		float left_edge = 
			info->camera_left + ((float) x / info->image_width) * info->camera_width;
		float right_edge = 
			info->camera_left + ((float) (x + 1) / info->image_width) * info->camera_width;
		float top_edge = 
			info->camera_top - ((float) y / info->image_height) * info->camera_height;
		float bottom_edge = 
			info->camera_top - ((float) (y + 1) / info->image_height) * info->camera_height;

		float r_x = left_edge + (right_edge - left_edge) * curand_uniform(&states[index]);
		float r_y = bottom_edge + (top_edge - bottom_edge) * curand_uniform(&states[index]);

		rays[index] = ray_create(vector3_create(0, 0, 0), vector3_create(r_x, r_y, 1));
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
	i->image_height = height;
	i->image_dim_ratio = (float) width / (float) height;
	i->camera_tan_half_fov = tanf(PI * fov / 360);
	i->camera_focus_plane = plane;
	i->camera_width = 2 * plane * i->camera_tan_half_fov;
	i->camera_height = i->camera_width / i->image_dim_ratio;
	i->camera_left = i->camera_width / -2;
	i->camera_top = i->camera_height / 2;
}

void call_kernel(Bitmap* bitmap)
{
	int N = bitmap->width * bitmap->height;
	int threads_per_block = 256;
	int blocks_amount = (N + threads_per_block - 1) / threads_per_block;

	RenderInfo info;
	init_render_info(&info, bitmap->width, bitmap->height, 90, 1);
	RenderInfo* d_info;
	cudaMalloc(&d_info, sizeof(RenderInfo));
	cudaMemcpy(d_info, &info, sizeof(RenderInfo), cudaMemcpyHostToDevice);

	curandState* d_states;
	cudaMalloc(&d_states, sizeof(curandState) * threads_per_block * blocks_amount);
	init_curand_states<<<blocks_amount, threads_per_block>>>(d_states, N);

	Ray* d_rays;
	cudaMalloc(&d_rays, sizeof(Ray) * N);

	Sphere* sphere = sphere_new(1, vector3_create(0, 0, 5));
	Sphere* d_sphere;
	cudaMalloc(&d_sphere, sizeof(Sphere));
	cudaMemcpy(d_sphere, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);

	Vector3 h_colors[N];
	for (int i = 0; i < N; i++)
	{
		h_colors[i] = vector3_create(0, 0, 0);
	}
	Vector3* d_colors;
	cudaMalloc(&d_colors, N * sizeof(Vector3));
	cudaMemcpy(d_colors, &h_colors, N * sizeof(Vector3), cudaMemcpyHostToDevice);

	for (int i = 0; i < 50; i++)
	{
		init_rays<<<blocks_amount, threads_per_block>>>(d_rays, d_info, d_states, N);
		pathtrace_kernel<<<blocks_amount, threads_per_block>>>(d_colors, d_rays, d_sphere, d_states, N);		
	}

	Pixel* h_pixels = bitmap->pixels;
	Pixel* d_pixels;
	cudaMalloc(&d_pixels, sizeof(Pixel) * N);
	cudaMemcpy(d_pixels, h_pixels, sizeof(Pixel) * N, cudaMemcpyHostToDevice);

	set_bitmap<<<blocks_amount, threads_per_block>>>(d_colors, d_pixels, 50, N);

	cudaMemcpy(h_pixels, d_pixels, sizeof(Pixel) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_colors);
	cudaFree(d_states);
	cudaFree(d_sphere);
	cudaFree(d_rays);
	cudaFree(d_info);
	cudaFree(d_pixels);

	sphere_free(sphere);
}