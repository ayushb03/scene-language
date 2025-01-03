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
                              rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0.15)) @
                              translation_matrix((0, 0, 0.15)))

    # Add minute hand
    minute_hand = primitive_call('cube', color=(0.1, 0.1, 0.1),
                               shape_kwargs={'scale': (0.05, size * 0.7, 0.05)})
    minute_hand = transform_shape(minute_hand,
                                rotation_matrix(math.pi/1.5, (0, 0, 1), (0, 0, 0.2)) @
                                translation_matrix((0, 0, 0.2)))

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
        center = (radius * math.cos(angle), i * step_size * 0.4, radius * math.sin(angle))
        return transform_shape(step,
                             rotation_matrix(angle, (0, 1, 0), center) @
                             translation_matrix(center))
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
                                        'p0': (0, 0, 0),
                                        'p1': (0, size, 0)})
    return concat_shapes(sphere, cube, cylinder)

@register()
def escher_scene() -> Shape:
    # Create main staircase
    stairs = library_call('infinite_staircase', num_steps=16, step_size=1.0)

    # Add floating clocks at different positions
    def clock_fn(i: int) -> Shape:
        clock = library_call('floating_clock', size=0.8)
        angle = (i * 2 * math.pi) / 5
        radius = 4
        pos = (radius * math.cos(angle), i * 1.5, radius * math.sin(angle))
        return transform_shape(clock, translation_matrix(pos))

    clocks = loop(5, clock_fn)

    # Add floating geometric shapes
    def geometry_fn(i: int) -> Shape:
        geom = library_call('floating_geometry', size=0.7)
        angle = (i * 2 * math.pi) / 4
        radius = 3
        pos = (radius * math.sin(angle), i * 2.0 - 2, radius * math.cos(angle))
        return transform_shape(geom,
                             rotation_matrix(angle, (1, 1, 1), pos) @
                             translation_matrix(pos))

    geometries = loop(4, geometry_fn)

    # Create reflections using a non-Euclidean mirror
    scene = concat_shapes(stairs, clocks, geometries)
    reflection = transform_shape(scene,
                               reflection_matrix((0, 0, 0), (1, 0.5, 0.5)) @
                               scale_matrix(0.7, (0, 0, 0)))

    return concat_shapes(scene, reflection)
"""

This implementation creates:

1. A clock system with face and hands
2. An infinite spiral staircase that winds upward
3. Floating geometric shapes (spheres, cubes, cylinders)
4. Multiple floating clocks at different positions
5. A reflection of the entire scene in a non-Euclidean mirror

The scene is built modularly:
- `clock_face` creates a clock base with hour markers
- `floating_clock` adds hands to the clock face
- `stair_step` creates a single stair step
- `infinite_staircase` creates a spiral staircase
- `floating_geometry` creates various geometric shapes
- `escher_scene` combines everything and adds the reflection

The scene has several Escher-like qualities:
- Impossible geometry through the infinite staircase
- Multiple perspectives and orientations
- Floating objects that defy gravity
- Non-Euclidean reflections that create a surreal effect
- Repetitive patterns that create visual rhythm

The colors and sizes are chosen to create visual contrast and depth, while the transformations create the illusion of impossible space characteristic of Escher's work.
"""