# illume
![Cornell Box](renders/cornell-1024x1024-100000spp-100md.png?raw=true "Title")
Original cornell box with 100,000 samples per pixel. 

## About
This is a unidirectional brute force pathtracer running on the GPU. 
### Features:
- GPU rendering
- materials: emissive, lambert, mirror, refractive, microfacet cook-torrance
- Monte Carlo subsurface scattering with Henyey-Greenstein phase function
- spheres and OBJ file meshes
- mesh instancing
- BVH construction and caching
- scene loading from human-readable JSON
- saves renders to PNG

### Dependencies/External Libraries
- CUDA 8.0
- Visual Studio 2015 (compilation)
- libPNG (PNG writing)
- rapidjson (JSON scene file parsing)

## More Renders
![sss dragon in box](renders/cornellsss-1024x1024-12500spp-50md.png?raw=true "Title")
Stanford dragon with orange absorption and foward scattering
![xyzrgb dragon](renders/xyzrgb2-960x720-20000spp-10md.png?raw=true "Title")
721k tri Stanford xyzrgb dragon, with Cook-Torrance teal surface
![Cornell Box](renders/box-960x720-25000spp-10md.png?raw=true "Title")
Cornell box with a glass ball and a mirror ball, showcasing reflection, refraction, caustics, and GI
![sss dragon](renders/hgdragonback4-1440x1080-4000spp-40md.png?raw=true "Title")
100k tri Stanford dragon model with moderate backscattering and a IOR of 1.68, making it kind of look like jade
![spheres](renders/spheres-1440x1080-15000spp-15md.png?raw=true "Title")
Replication in this renderer of the image on the wikipedia article for raytracing
![sss bunny](renders/sss-960x720-5000spp-45md.png?raw=true "Title")
Stanford bunny model with isotrophic scattering

## Papers/Links

http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

https://www.mitsuba-renderer.org/

http://photorealizer.blogspot.com/