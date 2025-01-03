"""
I'll help create a mesmerizing mandala pattern with geometric shapes. I'll break it down into modular components:

"""
from helper import *

"""
A perfectly symmetrical mandala composed of glowing geometric shapes, with every detail fractally mirrored and emanating light rays in a radial pattern.
"""

@register()
def petal(radius: float) -> Shape:
    """Creates a single petal using spheres"""
    sphere = primitive_call('sphere',
                          color=(0.9, 0.6, 0.9),
                          shape_kwargs={'radius': radius})
    return transform_shape(sphere,
                         translation_matrix((radius * 2, 0, 0)))

@register()
def petal_ring(radius: float, num_petals: int) -> Shape:
    """Creates a ring of petals"""
    def loop_fn(i: int) -> Shape:
        angle = (i * 2 * math.pi) / num_petals
        petal_shape = library_call('petal', radius=radius)
        center = (0, 0, 0)
        return transform_shape(petal_shape,
                             rotation_matrix(angle, (0, 0, 1), center))
    return loop(num_petals, loop_fn)

@register()
def ray(length: float, width: float) -> Shape:
    """Creates a single light ray using a cylinder"""
    return primitive_call('cylinder',
                        color=(1, 0.95, 0.8),
                        shape_kwargs={
                            'radius': width,
                            'p0': (0, 0, 0),
                            'p1': (length, 0, 0)
                        })

@register()
def ray_burst(num_rays: int, length: float, width: float) -> Shape:
    """Creates a radial burst of light rays"""
    def loop_fn(i: int) -> Shape:
        angle = (i * 2 * math.pi) / num_rays
        ray_shape = library_call('ray', length=length, width=width)
        center = (0, 0, 0)
        return transform_shape(ray_shape,
                             rotation_matrix(angle, (0, 0, 1), center))
    return loop(num_rays, loop_fn)

@register()
def mandala_layer(radius: float, num_elements: int, scale: float) -> Shape:
    """Creates one layer of the mandala"""
    ring = library_call('petal_ring', radius=radius, num_petals=num_elements)
    center = compute_shape_center(ring)
    scaled_ring = transform_shape(ring, scale_matrix(scale, center))
    return scaled_ring

@register()
def mandala() -> Shape:
    """Creates the complete mandala with multiple layers and light rays"""
    # Create multiple layers with different scales and rotations
    layer1 = library_call('mandala_layer', radius=0.1, num_elements=8, scale=1.0)
    layer2 = library_call('mandala_layer', radius=0.15, num_elements=12, scale=0.8)
    layer3 = library_call('mandala_layer', radius=0.2, num_elements=16, scale=0.6)

    # Create light rays
    rays1 = library_call('ray_burst', num_rays=24, length=0.4, width=0.01)
    rays2 = library_call('ray_burst', num_rays=16, length=0.3, width=0.015)

    # Rotate alternate layers for more intricate pattern
    layer2 = transform_shape(layer2,
                           rotation_matrix(math.pi/12, (0, 0, 1), (0, 0, 0)))
    layer3 = transform_shape(layer3,
                           rotation_matrix(math.pi/8, (0, 0, 1), (0, 0, 0)))

    # Combine all elements
    return concat_shapes(
        layer1,
        layer2,
        layer3,
        rays1,
        rays2
    )
"""

This program creates a beautiful mandala pattern with the following features:

1. Multiple layers of petal rings, each with different numbers of petals and scales
2. Each petal is created using spheres
3. Light rays emanating from the center in two layers:
   - Longer, thinner rays for outer burst
   - Shorter, slightly thicker rays for inner burst
4. Perfect symmetry achieved through precise angular calculations
5. Fractal-like appearance through scaled and rotated layers
6. Soft colors with glowing effect (light pink petals and warm yellow rays)

The mandala is built from the inside out, with:
- Inner layer: 8 petals
- Middle layer: 12 petals (rotated slightly)
- Outer layer: 16 petals (rotated slightly)
- Two layers of light rays with different densities and lengths

Each layer is precisely positioned and scaled relative to the center, creating a harmonious and balanced composition. The alternating rotations of the layers create additional visual interest and complexity while maintaining perfect symmetry.
"""