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
	float camera_dof;
	float camera_aperture;
	Vector3 camera_pos;
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
		RenderInfo i = *info;
		int x = index % (int) i.image_width;
		int y = index / (int) i.image_width;
		float left_edge = i.camera_left + i.camera_pixel_size * (float) x;
		float right_edge = left_edge + i.camera_pixel_size;
		float top_edge = i.camera_top - i.camera_pixel_size * (float) y;
		float bottom_edge = top_edge + i.camera_pixel_size;

		float r_x = left_edge + (right_edge - left_edge) * curand_uniform(&states[index]);
		float r_y = bottom_edge + (top_edge - bottom_edge) * curand_uniform(&states[index]);

		Vector3 pos;
		if (i.camera_aperture == 0)
		{
			pos = i.camera_pos;
		}
		else
		{
			float u1 = curand_uniform(&states[index]);
			float u2 = curand_uniform(&states[index]);
			pos = 
				vector3_add(vector3_mul(sample_circle(u1, u2), i.camera_aperture), i.camera_pos);
		}
		rays[index] = ray_create(pos, vector3_sub(vector3_create(r_x, r_y, i.camera_dof), pos));
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

	for (int i = 0; i < scene->mesh_amount; i++)
	{
		Intersection inter = mesh_ray_intersect(scene->meshes[i], ray);

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
	Vector3 sun = vector3_create(1, 1, -1);
	vector3_normalize(&sun);
	float grad = (vector3_dot(direction, sun) + 1) / 2;
	return vector3_add(vector3_mul(vector3_create(0.2, 0.2, 0.2), 1 - grad), 
					   vector3_mul(vector3_create(0.8, 0.8, 0.8), grad));
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
			if (min.m.type == MATERIAL_EMISSIVE)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
				vector3_add_to(&final_colors[ray_index], ray_colors[ray_index]);
				ray_statuses[index] = -1;
			}
			else if (min.m.type == MATERIAL_DIFFUSE)
			{
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
				Vector3 new_origin = ray_position_along(rays[ray_index], min.d);
				vector3_add_to(&new_origin, vector3_mul(min.normal, 10e-6));
				float u1 = curand_uniform(&states[ray_index]);
				float u2 = curand_uniform(&states[ray_index]);
				Vector3 sample = sample_hemisphere_cosine(u1, u2);
				Vector3 new_direction = vector3_to_basis(sample, min.normal);
				ray_set(&rays[ray_index], new_origin, new_direction);
			}
			else
			{
				Ray r = rays[ray_index];
				vector3_mul_vector_to(&ray_colors[ray_index], min.m.c);
				Vector3 new_origin = ray_position_along(r, min.d);
				vector3_add_to(&new_origin, vector3_mul(min.normal, 10e-6));
				Vector3 new_direction = vector3_reflect(r.d, min.normal);
				ray_set(&rays[ray_index], new_origin, new_direction);
			}
		}
		else
		{
			Vector3 sky = get_background_color(rays[ray_index].d);
			vector3_mul_vector_to(&ray_colors[ray_index], sky);
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
			*active_pixels = left;
		}
	}

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
	float tan_half_fov = tanf(PI * camera.fov / 360);
	i.camera_dof = camera.dof;	
	i.camera_aperture = camera.aperture;
	i.camera_pos = camera.pos;
	float dofmfov = i.camera_dof * tan_half_fov;
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
	Plane* d_planes;
	Mesh* d_meshes;
	int mesh_amount;
	Triangle** d_triangle_pointers;
} 
SceneReference;

static SceneReference allocate_scene_gpu(Scene* scene)
{
	SceneReference ref;
	int spheres_size = sizeof(Sphere) * scene->sphere_amount;
	int planes_size = sizeof(Plane) * scene->plane_amount;
	int meshes_size = sizeof(Mesh) * scene->mesh_amount;
	ref.mesh_amount = scene->mesh_amount;

	HANDLE_ERROR( cudaMalloc(&ref.d_scene, sizeof(Scene)) );
	HANDLE_ERROR( cudaMalloc(&ref.d_spheres, spheres_size) );
	HANDLE_ERROR( cudaMalloc(&ref.d_planes, planes_size) );
	HANDLE_ERROR( cudaMalloc(&ref.d_meshes, meshes_size) );
	ref.d_triangle_pointers = (Triangle **) calloc(scene->mesh_amount, sizeof(Triangle *));
	for (int i = 0; i < scene->mesh_amount; i++)
	{
		int triangles_size = scene->meshes[i].triangle_amount * sizeof(Triangle);
		HANDLE_ERROR( cudaMalloc(&ref.d_triangle_pointers[i], triangles_size) );
		HANDLE_ERROR( cudaMemcpy(
			ref.d_triangle_pointers[i], scene->meshes[i].triangles, triangles_size, cudaMemcpyHostToDevice) );
	}

	Triangle** h_triangle_pointers = (Triangle **) calloc(scene->mesh_amount, sizeof(Triangle *));
	for (int i = 0; i < scene->mesh_amount; i++)
	{
		h_triangle_pointers[i] = scene->meshes[i].triangles;
		scene->meshes[i].triangles = ref.d_triangle_pointers[i];	
	}

	Sphere* h_spheres = scene->spheres;
	Plane* h_planes = scene->planes;
	Mesh* h_meshes = scene->meshes;
	scene->spheres = ref.d_spheres;
	scene->planes = ref.d_planes;
	scene->meshes = ref.d_meshes;
	HANDLE_ERROR( cudaMemcpy(ref.d_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice) );
	scene->spheres = h_spheres;
	scene->planes = h_planes;
	scene->meshes = h_meshes;
	HANDLE_ERROR( cudaMemcpy(ref.d_spheres, scene->spheres, spheres_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(ref.d_planes, scene->planes, planes_size, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(ref.d_meshes, scene->meshes, meshes_size, cudaMemcpyHostToDevice) );
	for (int i = 0; i < scene->mesh_amount; i++)
	{
		scene->meshes[i].triangles = h_triangle_pointers[i];
	}
	free(h_triangle_pointers);
	return ref;
}

static void free_scene_gpu(SceneReference ref)
{
	HANDLE_ERROR( cudaFree(ref.d_spheres) );
	HANDLE_ERROR( cudaFree(ref.d_planes) );
	HANDLE_ERROR( cudaFree(ref.d_meshes) );
	HANDLE_ERROR( cudaFree(ref.d_scene) );
	for (int i = 0; i < ref.mesh_amount; i++)
	{
		HANDLE_ERROR( cudaFree(ref.d_triangle_pointers[i]) );
	}
	free(ref.d_triangle_pointers);
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

void render_scene(Scene* scene, Bitmap* bitmap, Camera camera, int samples, int max_depth)
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

	RenderInfo* d_info = 
		allocate_render_info_gpu(bitmap->width, bitmap->height, camera);

	Vector3* d_final_colors = allocate_final_colors_gpu(pixels_amount);

	Vector3* d_ray_colors;
	HANDLE_ERROR( cudaMalloc(&d_ray_colors, pixels_amount * sizeof(Vector3)) );

	int* d_ray_statuses;
	HANDLE_ERROR( cudaMalloc(&d_ray_statuses, pixels_amount * sizeof(int)) );

	Ray* d_rays;
	HANDLE_ERROR( cudaMalloc(&d_rays, sizeof(Ray) * pixels_amount) );

	SceneReference ref = allocate_scene_gpu(scene);

	int* h_ray_statuses = (int *) calloc(pixels_amount, sizeof(int));

	cudaEvent_t calc_start;
	cudaEvent_t calc_stop;
	start_timer(&calc_start, &calc_stop);

	for (int i = 0; i < samples; i++)
	{
		init_rays<<<blocks_amount, threads_per_block>>>
			(d_rays, d_ray_statuses, d_ray_colors, d_info, d_states, pixels_amount);

		int active_pixels = pixels_amount;
		int blocks = blocks_amount;

		for (int j = 0; j < max_depth; j++)
		{
			pathtrace_kernel<<<blocks, threads_per_block>>>
				(d_final_colors, d_rays, d_ray_statuses, d_ray_colors, 
				 ref.d_scene, d_states, active_pixels);		

			compact_pixels(d_ray_statuses, h_ray_statuses, &active_pixels);
			blocks = (active_pixels + threads_per_block - 1) / threads_per_block;
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
	free(h_ray_statuses);

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