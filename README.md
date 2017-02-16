# illume
![Cornell Box](renders/cornell-1024x1024-100000spp-100md.png?raw=true "Title")
Original cornell box with 100,000 samples per pixel. 

## About
This is a unidirectional brute force pathtracer running on the GPU. 
### Features:
- GPU rendering with CUDA
- materials: emissive, lambert, mirror, refractive, microfacet reflection and refraction,
			 smooth and rough conductor
- textures: constant, checkerboard, and bitmap
- HDR environment maps
- Tonemapping operators: linear, Reinhard, filmic, and Uncharted2
- Monte Carlo subsurface scattering with Henyey-Greenstein phase function
- primitives: spheres, trimeshes
- OBJ loading
- mesh instancing
- BVH construction and caching
- scene loading from human-readable JSON

### Dependencies/External Libraries
- CUDA 8.0
- Visual Studio 2015 (compilation)
- libPNG (PNG writing)
- rapidjson (JSON scene file parsing)

## More Renders
![sss dragon in box](renders/cornellsss-1024x1024-12500spp-50md.png?raw=true "Title")
Stanford dragon with orange absorption and foward scattering

![rough glass](renders/lucy-rough-dielectric-1024x1024-70000spp-25md.png?raw=true "")
525k Stanford lucy with ground glass

![conductors](renders/conductor-1024x1024-25000spp-25md.png?raw=true "")
Rough gold lucy and rough copper Stanford dragon

![xyzrgb dragon](renders/xyzrgb2-960x720-20000spp-10md.png?raw=true "Title")
721k tri Stanford xyzrgb dragon, with Cook-Torrance teal surface

![Cornell Box](renders/box-960x720-25000spp-10md.png?raw=true "Title")
Cornell box with a glass ball and a mirror ball, showcasing reflection, refraction, caustics, and GI

![spheres](renders/spheres-1440x1080-15000spp-15md.png?raw=true "Title")
Replication in this renderer of the image on the wikipedia article for raytracing

![sss bunny](renders/sss-960x720-5000spp-45md.png?raw=true "Title")
Stanford bunny model with isotrophic scattering

## Papers/Links

Reflection/refraction, fresnel: http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

Rough Dielectric: https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

Complex iors: refractiveindex.info/?shelf=3d&book=metals

Conductor fresnel equations: seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/#more-1921

Tonemapping: http://filmicworlds.com/blog/filmic-tonemapping-operators/

https://www.mitsuba-renderer.org/

http://photorealizer.blogspot.com/