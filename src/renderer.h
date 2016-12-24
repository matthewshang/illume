#pragma once

#include "bitmap.h"
#include "scene/scene.h"

class Renderer
{
public:
	Renderer(Scene* scene, int spp, int max_depth);
	void render_to_bitmap(Bitmap* bitmap);

private:
	Scene* m_scene;
	int m_spp;
	int m_max_depth;
};