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
def ray(length: float, color: tuple[float, float, float]) -> Shape:
    """Creates a light ray using a thin cylinder"""
    return primitive_call('cylinder',
                        shape_kwargs={
                            'radius': 0.02,
                            'p0': (0, 0, 0),
                            'p1': (length, 0, 0)
                        },
                        color=color)

@register()
def ray_burst(num_rays: int, length: float, color: tuple[float, float, float]) -> Shape:
    """Creates a burst of light rays"""
    def loop_fn(i: int) -> Shape:
        angle = (i * 2 * math.pi) / num_rays
        ray_shape = library_call('ray', length=length, color=color)
        return transform_shape(ray_shape,
                             rotation_matrix(angle, (0, 0, 1), (0, 0, 0)))

    return loop(num_rays, loop_fn)

@register()
def mandala() -> Shape:
    # Colors for different layers
    inner_color = (1.0, 0.7, 0.2)  # warm golden
    mid_color = (0.8, 0.3, 0.9)    # purple
    outer_color = (0.3, 0.8, 1.0)  # light blue
    ray_color = (1.0, 0.95, 0.8)   # light yellow

    # Build from inside out
    center = library_call('center_piece')

    # Inner ring
    inner_ring = library_call('petal_ring',
                            radius=0.3,
                            num_petals=8,
                            petal_size=0.1,
                            color=inner_color)

    # Middle ring
    middle_ring = library_call('petal_ring',
                             radius=0.6,
                             num_petals=12,
                             petal_size=0.12,
                             color=mid_color)

    # Outer ring
    outer_ring = library_call('petal_ring',
                            radius=0.9,
                            num_petals=16,
                            petal_size=0.14,
                            color=outer_color)

    # Light rays
    rays = library_call('ray_burst',
                       num_rays=24,
                       length=1.2,
                       color=ray_color)

    # Combine all elements
    return concat_shapes(
        rays,          # bottom layer
        outer_ring,    # then outer elements
        middle_ring,   # then middle elements
        inner_ring,    # then inner elements
        center         # center on top
    )
"""

This program creates a beautiful mandala pattern with the following features:

1. A glowing central sphere in golden color
2. Three concentric rings of spherical "petals":
   - Inner ring: 8 golden petals
   - Middle ring: 12 purple petals
   - Outer ring: 16 blue petals
3. A burst of 24 light rays emanating from the center, created using thin cylinders

The design follows these principles:
- Perfect radial symmetry using precise mathematical positioning
- Fractal-like scaling with increasing number of elements in outer rings
- Harmonious color progression from warm center to cool outer elements
- Layered composition with light rays underneath and center sphere on top
- All transformations are calculated using exact trigonometric functions for perfect alignment

The mandala can be created by simply calling `library_call('mandala')`. Each component is modular and can be easily modified by adjusting parameters like colors, sizes, and number of elements in each ring.
"""