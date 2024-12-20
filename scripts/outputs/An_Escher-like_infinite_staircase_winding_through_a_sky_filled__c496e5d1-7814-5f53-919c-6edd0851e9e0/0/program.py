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
        angle = (i * 2 * math.pi) / 12
        center = (radius * 0.8 * math.cos(angle), radius * 0.8 * math.sin(angle), 0.05)
        return transform_shape(marker,
                             rotation_matrix(angle, (0, 0, 1), center) @
                             translation_matrix(center))

    return concat_shapes(face, loop(12, marker_fn))

@register()
def floating_clock(size: float) -> Shape:
    # Create complete clock with hands
    clock = library_call('clock_face', radius=size)

    # Add hour hand
    hour_hand = primitive_call('cube', color=(0.1, 0.1, 0.1),
                             shape_kwargs={'scale': (0.08, size * 0.5, 0.08)})
    hour_hand = transform_shape(hour_hand,
                              rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0)) @
                              translation_matrix((0, size * 0.25, 0.15)))

    # Add minute hand
    minute_hand = primitive_call('cube', color=(0.1, 0.1, 0.1),
                               shape_kwargs={'scale': (0.05, size * 0.7, 0.05)})
    minute_hand = transform_shape(minute_hand,
                                rotation_matrix(math.pi/1.5, (0, 0, 1), (0, 0, 0)) @
                                translation_matrix((0, size * 0.35, 0.15)))

    return concat_shapes(clock, hour_hand, minute_hand)

@register()
def stair_step(width: float, height: float, depth: float) -> Shape:
    return primitive_call('cube', color=(0.7, 0.7, 0.8),
                        shape_kwargs={'scale': (width, height, depth)})

@register()
def infinite_staircase(num_steps: int, step_size: float) -> Shape:
    def step_fn(i: int) -> Shape:
        step = library_call('stair_step', width=step_size, height=step_size*0.5, depth=step_size)
        angle = (i * 2 * math.pi) / num_steps
        radius = step_size * 3

        # Create spiral effect
        pos = (radius * math.cos(angle),
               i * step_size * 0.4,  # Gradual elevation
               radius * math.sin(angle))

        return transform_shape(step,
                             rotation_matrix(angle, (0, 1, 0), pos) @
                             translation_matrix(pos))

    return loop(num_steps, step_fn)

@register()
def floating_geometry(size: float) -> Shape:
    # Create various geometric shapes
    sphere = primitive_call('sphere', color=(0.9, 0.5, 0.5),
                          shape_kwargs={'radius': size*0.5})
    cube = primitive_call('cube', color=(0.5, 0.9, 0.5),
                         shape_kwargs={'scale': (size, size, size)})
    cylinder = primitive_call('cylinder', color=(0.5, 0.5, 0.9),
                            shape_kwargs={'radius': size*0.3,
                                        'p0': (0, -size/2, 0),
                                        'p1': (0, size/2, 0)})

    # Position them in space
    cube = transform_shape(cube, translation_matrix((size*2, 0, 0)))
    cylinder = transform_shape(cylinder, translation_matrix((-size*2, 0, 0)))

    return concat_shapes(sphere, cube, cylinder)

@register()
def escher_scene() -> Shape:
    # Create main staircase
    stairs = library_call('infinite_staircase', num_steps=16, step_size=1.0)

    # Create floating clocks at different positions
    def clock_fn(i: int) -> Shape:
        clock = library_call('floating_clock', size=0.8)
        angle = (i * 2 * math.pi) / 5
        radius = 4
        pos = (radius * math.cos(angle),
               2 + math.sin(i * 1.5),  # Varying heights
               radius * math.sin(angle))

        return transform_shape(clock,
                             rotation_matrix(angle, (1, 1, 1), pos) @
                             translation_matrix(pos))

    clocks = loop(5, clock_fn)

    # Add geometric shapes
    def geometry_fn(i: int) -> Shape:
        geom = library_call('floating_geometry', size=0.7)
        angle = (i * 2 * math.pi) / 3
        radius = 6
        pos = (radius * math.cos(angle),
               3 + math.cos(i * 2),  # Varying heights
               radius * math.sin(angle))

        return transform_shape(geom,
                             rotation_matrix(angle * 2, (0, 1, 0), pos) @
                             translation_matrix(pos))

    geometries = loop(3, geometry_fn)

    # Create mirrored version
    scene = concat_shapes(stairs, clocks, geometries)
    mirror_plane = reflection_matrix((0, 0, 0), (1, 1, 1))
    mirrored_scene = transform_shape(scene, mirror_plane)

    return concat_shapes(scene, mirrored_scene)
"""

This program creates:
1. A clock face with hour markers and hands
2. An infinite spiral staircase that winds upward
3. Floating geometric shapes (spheres, cubes, cylinders)
4. Multiple floating clocks at different angles and positions
5. A reflection of the entire scene across an invisible mirror plane

The scene has several key features:
- The staircase creates an impossible geometry effect by spiraling upward continuously
- Clocks float at different heights and angles, adding to the surreal atmosphere
- Geometric shapes are positioned in a circular pattern around the scene
- Everything is reflected across a diagonal mirror plane, creating a non-Euclidean feel
- The color scheme uses soft, pastel colors for a dreamlike quality

The scene is modular and parameterized, allowing for easy adjustments to:
- Number of stairs and their size
- Number and size of clocks
- Position and scale of geometric shapes
- The reflection plane orientation
"""