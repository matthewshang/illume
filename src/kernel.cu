#include "kernel.h"

__global__ 
void init_curand_states(curandState* states, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		curand_init((666420691337 << 20) + index, 0, 0, &states[index]);
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
static Intersection get_min_intersection(Scene* scene, Ray ray)
{
	Intersection min = intersection_create_no_intersect();
	min.d = FLT_MAX;
	for (int i = 0; i < scene->sphere_amount; i++)
	{
		Intersection inter = sphere_ray_intersect(scene->spheres[i], ray);

		if (inter.is_intersect == 1 && inter.d < min.d)
		{
			min = inter;
		}
	}
	
	for (int i = 0; i < scene->plane_amount; i++)
	{
		Intersection inter = plane_ray_intersect(scene->planes[i], ray);

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
	float grad = (direction.x + 1) / 2;
	return vector3_create(grad, grad, grad);
}

__global__
void pathtrace_kernel(Vector3* final_colors, Ray* rays, int* ray_statuses, 
					  Vector3* ray_colors, Scene* scene, curandState* states, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int ray_index = ray_statuses[index];
	if (index < N && ray_index != -1)
	{
		Intersection min = get_min_intersection(scene, rays[ray_index]);
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
				Vector3 bias = vector3_mul(min.normal, 10e-6);
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
		float gamma = 1 / 2.2;
		Vector3 corrected = vector3_mul(final_colors[index], 1 / samples);
		corrected = vector3_max(vector3_min(corrected, 1), 0);
		corrected = vector3_pow(corrected, gamma);
		pixels[index].red = (int) (255 * corrected.x);
		pixels[index].green = (int) (255 * corrected.y);
		pixels[index].blue = (int) (255 * corrected.z);
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
	Plane* d_planes;
} 
SceneReference;

static SceneReference allocate_scene_gpu(Scene* scene)
{
	SceneReference ref;
	int spheres_size = sizeof(Sphere) * scene->sphere_amount;
	int planes_size = sizeof(Plane) * scene->plane_amount;

	HANDLE_ERROR( cudaMalloc(&ref.d_scene, sizeof(Scene)) );
	HANDLE_ERROR( cudaMalloc(&ref.d_spheres, spheres_size) );
	HANDLE_ERROR( cudaMalloc(&ref.d_planes, planes_size) );
	Sphere* h_spheres = scene->spheres;
	Plane* h_planes = scene->planes;
	scene->spheres = ref.d_spheres;
	scene->planes = ref.d_planes;
	HANDLE_ERROR( cudaMemcpy(ref.d_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice) );
	scene->spheres = h_spheres;
	scene->planes = h_planes;
	HANDLE_ERROR( cudaMemcpy(ref.d_spheres, scene->spheres, spheres_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(ref.d_planes, scene->planes, planes_size, cudaMemcpyHostToDevice) );
	
	return ref;
}

static void free_scene_gpu(SceneReference ref)
{
	HANDLE_ERROR( cudaFree(ref.d_spheres) );
	HANDLE_ERROR( cudaFree(ref.d_scene) );
	HANDLE_ERROR( cudaFree(ref.d_planes) );
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

void render_scene(Scene* scene, Bitmap* bitmap, int samples, int max_depth)
{
	if (!scene)
	{
		return;
	}

	cudaEvent_t render_start;
	cudaEvent_t render_stop;
	start_timer(&render_start, &render_stop);

	HANDLE_ERROR( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024) );
	int pixels_amount = bitmap->width * bitmap->height;
	int threads_per_block = 256;
	int blocks_amount = (pixels_amount + threads_per_block - 1) / threads_per_block;

	curandState* d_states;
	HANDLE_ERROR( cudaMalloc(&d_states, sizeof(curandState) * threads_per_block * blocks_amount) );
	init_curand_states<<<blocks_amount, threads_per_block>>>(d_states, pixels_amount);

	RenderInfo* d_info = allocate_render_info_gpu(bitmap->width, bitmap->height, 70, 1);

	Vector3* d_final_colors = allocate_final_colors_gpu(pixels_amount);

	Vector3* d_ray_colors;
	HANDLE_ERROR( cudaMalloc(&d_ray_colors, pixels_amount * sizeof(Vector3)) );

	int* d_ray_statuses;
	HANDLE_ERROR( cudaMalloc(&d_ray_statuses, pixels_amount * sizeof(int)) );

	Ray* d_rays;
	HANDLE_ERROR( cudaMalloc(&d_rays, sizeof(Ray) * pixels_amount) );

	SceneReference ref = allocate_scene_gpu(scene);

	cudaEvent_t calc_start;
	cudaEvent_t calc_stop;
	start_timer(&calc_start, &calc_stop);

	for (int i = 0; i < samples; i++)
	{
		init_rays<<<blocks_amount, threads_per_block>>>
			(d_rays, d_ray_statuses, d_ray_colors, d_info, d_states, pixels_amount);

		for (int j = 0; j < max_depth; j++)
		{
			pathtrace_kernel<<<blocks_amount, threads_per_block>>>
				(d_final_colors, d_rays, d_ray_statuses, d_ray_colors, 
				 ref.d_scene, d_states, pixels_amount);		
		}
	}

	float calc_time;
	end_timer(&calc_start, &calc_stop, &calc_time);

	HANDLE_ERROR( cudaFree(d_states) );
	HANDLE_ERROR( cudaFree(d_rays) );
	HANDLE_ERROR( cudaFree(d_info) );
	HANDLE_ERROR( cudaFree(d_ray_statuses) );
	HANDLE_ERROR( cudaFree(d_ray_colors) );
	free_scene_gpu(ref);

	Pixel* h_pixels = bitmap->pixels;
	Pixel* d_pixels;
	HANDLE_ERROR( cudaMalloc(&d_pixels, sizeof(Pixel) * pixels_amount) );
	HANDLE_ERROR( cudaMemcpy(d_pixels, h_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyHostToDevice) );

	set_bitmap<<<blocks_amount, threads_per_block>>>(d_final_colors, d_pixels, (float) samples, pixels_amount);
	HANDLE_ERROR( cudaMemcpy(h_pixels, d_pixels, sizeof(Pixel) * pixels_amount, cudaMemcpyDeviceToHost) );

	HANDLE_ERROR( cudaFree(d_final_colors) );
	HANDLE_ERROR( cudaFree(d_pixels) );

	float render_time;
	end_timer(&render_start, &render_stop, &render_time);

	printf("Calculation time: %f seconds\n", 1e-3 * (double) calc_time);
	printf("Render time: %f seconds\n", 1e-3 * (double) render_time);
}