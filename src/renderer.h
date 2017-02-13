#pragma once

#include "rapidjson/document.h"

#include "bitmap.h"
#include "scene/scene.h"

class Renderer
{
public:
	Renderer(rapidjson::Value& json, HostScene& scene, int spp, int max_depth);
	void render_to_bitmap(Bitmap* bitmap);

	int get_width();
	int get_height();

private:
	HostScene& m_scene;
	int m_width;
	int m_height;
	float m_ray_bias;
	int m_spp;
	int m_max_depth;
};