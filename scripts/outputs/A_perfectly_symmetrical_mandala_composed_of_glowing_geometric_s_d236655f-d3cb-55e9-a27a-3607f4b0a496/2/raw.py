"""
I'll help create a mesmerizing mandala pattern using geometric primitives. Here's a solution that builds the pattern from inside out:

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
                        color=(1.0, 0.9, 0.3))  # golden center

@register()
def petal(size: float, color: tuple[float, float, float]) -> Shape:
    """Creates a single petal using scaled spheres"""
    return primitive_call('sphere',
                        shape_kwargs={'radius': size},
                        color=color)

@register()
def petal_ring(radius: float, num_petals: int, petal_size: float, color: tuple[float, float, float]) -> Shape:
    """Creates a ring of petals at given radius"""
    def loop_fn(i: int) -> Shape:
        angle = (i * 2 * math.pi) / num_petals
        # Position petal
        pos = (radius * math.cos(angle), radius * math.sin(angle), 0)
        petal_shape = library_call('petal', size=petal_size, color=color)
        return transform_shape(petal_shape, translation_matrix(pos))

    return loop(num_petals, loop_fn)

@register()
def light_ray(length: float, width: float) -> Shape:
    """Creates a single light ray using a cylinder"""
    return primitive_call('cylinder',
                        shape_kwargs={
                            'radius': width,
                            'p0': (0, 0, 0),
                            'p1': (length, 0, 0)
                        },
                        color=(1.0, 0.95, 0.8))

@register()
def ray_ring(radius: float, num_rays: int, ray_length: float, ray_width: float) -> Shape:
    """Creates a ring of light rays"""
    def loop_fn(i: int) -> Shape:
        angle = (i * 2 * math.pi) / num_rays
        ray = library_call('light_ray', length=ray_length, width=ray_width)
        # Rotate and position each ray
        transform = translation_matrix((radius * math.cos(angle), radius * math.sin(angle), 0))
        rotation = rotation_matrix(angle, (0, 0, 1), (0, 0, 0))
        return transform_shape(ray, transform @ rotation)

    return loop(num_rays, loop_fn)

@register()
def mandala() -> Shape:
    # Build from inside out
    center = library_call('center_piece')

    # Inner petal rings
    ring1 = library_call('petal_ring', radius=0.3, num_petals=8, petal_size=0.08,
                        color=(0.9, 0.4, 0.4))  # red
    ring2 = library_call('petal_ring', radius=0.5, num_petals=12, petal_size=0.1,
                        color=(0.4, 0.6, 0.9))  # blue
    ring3 = library_call('petal_ring', radius=0.7, num_petals=16, petal_size=0.12,
                        color=(0.9, 0.5, 0.9))  # purple

    # Light rays
    rays1 = library_call('ray_ring', radius=0.2, num_rays=16, ray_length=0.15, ray_width=0.02)
    rays2 = library_call('ray_ring', radius=0.4, num_rays=24, ray_length=0.2, ray_width=0.02)
    rays3 = library_call('ray_ring', radius=0.6, num_rays=32, ray_length=0.25, ray_width=0.02)

    # Combine all elements
    return concat_shapes(
        center,
        rays1,
        ring1,
        rays2,
        ring2,
        rays3,
        ring3
    )
"""

This program creates a beautiful mandala pattern with the following features:

1. A glowing golden sphere at the center
2. Three rings of petals at different radii, with increasing numbers of petals and sizes
3. Three rings of light rays between the petal rings, creating a radial pattern
4. Each ring uses different colors to create visual interest
5. Perfect symmetry is maintained through precise mathematical positioning using trigonometry
6. The pattern grows larger and more complex as it radiates outward

The mandala is built using these key components:
- `center_piece`: Creates the central golden sphere
- `petal`: Defines a single petal shape using a sphere
- `petal_ring`: Creates a ring of evenly-spaced petals
- `light_ray`: Creates a single light ray using a cylinder
- `ray_ring`: Creates a ring of evenly-spaced light rays
- `mandala`: Combines all elements with proper positioning

The design maintains perfect symmetry by using mathematical calculations to position elements and consistent spacing between components. The light rays and colored petals create a sense of radiance and spiritual energy typical of mandala designs.
"""