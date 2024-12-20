"""
I'll create an abstract, Escher-inspired scene with infinite stairs, floating clocks, and geometric shapes with reflections. Here's my implementation:

"""
from helper import *

@register()
def clock_face(radius: float) -> Shape:
    # Create clock base
    face = primitive_call('cylinder', color=(0.9, 0.9, 0.9),
                         shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, 0, 0.1)})

    # Create hour markers
    def marker_fn(i: int) -> Shape:
        marker = primitive_call('cube', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'scale': (0.1, 0.4, 0.1)})
        angle = (i / 12) * 2 * math.pi
        pos = (radius * 0.8 * math.cos(angle), radius * 0.8 * math.sin(angle), 0.1)
        return transform_shape(marker,
                             rotation_matrix(angle, (0, 0, 1), pos) @ translation_matrix(pos))

    return concat_shapes(face, loop(12, marker_fn))

@register()
def floating_clock(radius: float) -> Shape:
    clock = library_call('clock_face', radius=radius)
    # Add clock hands
    hour_hand = primitive_call('cube', color=(0.1, 0.1, 0.1),
                             shape_kwargs={'scale': (0.08, radius * 0.5, 0.05)})
    minute_hand = primitive_call('cube', color=(0.1, 0.1, 0.1),
                               shape_kwargs={'scale': (0.05, radius * 0.7, 0.05)})

    hour_hand = transform_shape(hour_hand, rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0.15)))
    minute_hand = transform_shape(minute_hand, rotation_matrix(math.pi/2, (0, 0, 1), (0, 0, 0.2)))

    return concat_shapes(clock, hour_hand, minute_hand)

@register()
def stair_step() -> Shape:
    horizontal = primitive_call('cube', color=(0.7, 0.7, 0.8),
                              shape_kwargs={'scale': (1, 0.2, 1)})
    vertical = primitive_call('cube', color=(0.65, 0.65, 0.75),
                            shape_kwargs={'scale': (1, 1, 0.2)})
    vertical = transform_shape(vertical, translation_matrix((0, 0.6, -0.4)))
    return concat_shapes(horizontal, vertical)

@register()
def infinite_staircase(num_steps: int) -> Shape:
    def step_fn(i: int) -> Shape:
        step = library_call('stair_step')
        rotation = rotation_matrix(i * math.pi/2, (0, 1, 0), (0, 0, 0))
        translation = translation_matrix((0, i * 0.2, 0))
        scale = scale_matrix(0.95 ** i, (0, 0, 0))
        return transform_shape(step, rotation @ translation @ scale)

    return loop(num_steps, step_fn)

@register()
def floating_geometry(size: float) -> Shape:
    shapes = []
    # Create platonic solids approximations
    tetrahedron = primitive_call('sphere', color=(0.9, 0.3, 0.3),
                                shape_kwargs={'radius': size * 0.5})
    cube = primitive_call('cube', color=(0.3, 0.9, 0.3),
                         shape_kwargs={'scale': (size, size, size)})
    octahedron = primitive_call('sphere', color=(0.3, 0.3, 0.9),
                               shape_kwargs={'radius': size * 0.6})

    shapes.extend([tetrahedron, cube, octahedron])
    return concat_shapes(*shapes)

@register()
def escher_scene() -> Shape:
    # Create main staircase
    stairs = library_call('infinite_staircase', num_steps=12)

    # Add floating clocks at different positions
    def clock_fn(i: int) -> Shape:
        clock = library_call('floating_clock', radius=0.5)
        angle = (i / 5) * 2 * math.pi
        pos = (3 * math.cos(angle), 2 + i * 0.5, 3 * math.sin(angle))
        rotation = rotation_matrix(angle, (0, 1, 0), pos)
        return transform_shape(clock, translation_matrix(pos) @ rotation)

    clocks = loop(5, clock_fn)

    # Add floating geometric shapes
    def geometry_fn(i: int) -> Shape:
        geom = library_call('floating_geometry', size=0.4)
        angle = (i / 3) * 2 * math.pi
        pos = (2 * math.sin(angle), 1 + i * 0.8, 2 * math.cos(angle))
        rotation = rotation_matrix(i * math.pi/3, (1, 1, 1), pos)
        return transform_shape(geom, translation_matrix(pos) @ rotation)

    geometries = loop(3, geometry_fn)

    # Create reflections using a non-Euclidean mirror
    scene = concat_shapes(stairs, clocks, geometries)
    reflection = transform_shape(scene,
                               reflection_matrix((0, 0, 0), (1, 0.5, 0.5)) @
                               scale_matrix(0.8, (0, 0, 0)))

    return concat_shapes(scene, reflection)
"""

This program creates:

1. A clock face with hour markers and hands
2. An infinite staircase that spirals upward while scaling down
3. Floating clocks at different positions and orientations
4. Floating geometric shapes (approximations of platonic solids)
5. A reflection of the entire scene in a non-Euclidean mirror

The scene has several key features:
- The staircase creates an impossible geometry effect by spiraling upward while scaling down
- Clocks float at different heights and angles, creating a surreal atmosphere
- Geometric shapes add visual interest and depth
- The entire scene is reflected in a tilted, non-Euclidean mirror that slightly scales down the reflection
- The color palette uses subtle grays for the stairs and bright colors for the geometric shapes

The composition creates an Escher-like impossible space with multiple focal points and a dreamlike quality through the floating elements and their reflections.
"""