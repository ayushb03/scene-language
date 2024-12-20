"""
I'll help create a mesmerizing mandala pattern using geometric primitives. Here's a solution that builds the pattern from inside out with radial symmetry:

"""
from helper import *

"""
A perfectly symmetrical mandala composed of glowing geometric shapes, with every detail fractally mirrored and emanating light rays in a radial pattern.
"""

@register()
def center_piece() -> Shape:
    """Creates the central sphere of the mandala"""
    return primitive_call('sphere',
                        shape_kwargs={'radius': 0.15},
                        color=(1.0, 0.9, 0.4))  # golden center

@register()
def petal(size: float) -> Shape:
    """Creates a single petal using scaled spheres"""
    return primitive_call('sphere',
                        shape_kwargs={'radius': size},
                        color=(0.8, 0.4, 0.9))  # purple glow

@register()
def ray(length: float) -> Shape:
    """Creates a light ray using a thin cylinder"""
    return primitive_call('cylinder',
                        shape_kwargs={
                            'radius': 0.02,
                            'p0': (0, 0, 0),
                            'p1': (0, length, 0)
                        },
                        color=(1.0, 0.95, 0.8))  # light ray color

@register()
def petal_ring(radius: float, num_petals: int, petal_size: float) -> Shape:
    """Creates a ring of petals"""
    def create_petal(i: int) -> Shape:
        angle = (2 * math.pi * i) / num_petals
        petal_shape = library_call('petal', size=petal_size)
        # Position petal
        transform = translation_matrix((radius * math.cos(angle),
                                     radius * math.sin(angle),
                                     0))
        return transform_shape(petal_shape, transform)

    return loop(num_petals, create_petal)

@register()
def ray_ring(radius: float, num_rays: int, ray_length: float) -> Shape:
    """Creates a ring of light rays"""
    def create_ray(i: int) -> Shape:
        angle = (2 * math.pi * i) / num_rays
        ray_shape = library_call('ray', length=ray_length)
        # Position and rotate ray
        transform = translation_matrix((radius * math.cos(angle),
                                     radius * math.sin(angle),
                                     0))
        rotation = rotation_matrix(angle, (0, 0, 1), (0, 0, 0))
        return transform_shape(ray_shape, transform @ rotation)

    return loop(num_rays, create_ray)

@register()
def mandala() -> Shape:
    """Assembles the complete mandala with multiple layers"""
    # Start with center
    center = library_call('center_piece')

    # Create multiple rings of petals with decreasing sizes
    inner_petals = library_call('petal_ring', radius=0.3, num_petals=8, petal_size=0.1)
    middle_petals = library_call('petal_ring', radius=0.6, num_petals=12, petal_size=0.08)
    outer_petals = library_call('petal_ring', radius=0.9, num_petals=16, petal_size=0.06)

    # Create light rays at different radii
    inner_rays = library_call('ray_ring', radius=0.45, num_rays=16, ray_length=0.2)
    outer_rays = library_call('ray_ring', radius=0.75, num_rays=24, ray_length=0.3)

    # Combine all elements
    return concat_shapes(
        center,
        inner_petals,
        middle_petals,
        outer_petals,
        inner_rays,
        outer_rays
    )
"""

This program creates a beautiful mandala pattern with the following features:

1. A glowing central sphere in golden color
2. Three concentric rings of petals with:
   - Different numbers of petals (8, 12, 16)
   - Decreasing petal sizes as they go outward
   - Purple-tinted color for a mystical appearance

3. Two rings of light rays:
   - Between the petal rings
   - Different densities (16 and 24 rays)
   - Varying lengths for visual interest

4. Perfect radial symmetry achieved through:
   - Precise angular calculations
   - Consistent spacing
   - Careful transformation matrices

The mandala is constructed layer by layer, with each element precisely positioned using transformation matrices. The use of loops ensures perfect symmetry, while the varying sizes and numbers of elements create a sense of depth and complexity.
"""