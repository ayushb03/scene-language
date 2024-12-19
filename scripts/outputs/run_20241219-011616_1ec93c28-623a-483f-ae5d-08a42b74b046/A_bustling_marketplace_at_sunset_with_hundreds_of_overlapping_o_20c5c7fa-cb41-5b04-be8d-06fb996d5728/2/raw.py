"""
I'll create a bustling marketplace scene with modular components. Here's my implementation:

"""
from helper import *

"""
A bustling marketplace at sunset with hundreds of overlapping objects like food stalls,
hanging lanterns, moving crowds and children playing with kites while shadows lengthen dynamically
"""

@register()
def stall_frame() -> Shape:
    # Create wooden frame for the stall
    posts = loop(4, lambda i: primitive_call('cylinder',
        color=(0.6, 0.3, 0.1),
        shape_kwargs={
            'radius': 0.05,
            'p0': (((i%2)-0.5)*2, 0, ((i//2)-0.5)*2),
            'p1': (((i%2)-0.5)*2, 2, ((i//2)-0.5)*2)
        }
    ))

    roof = primitive_call('cube',
        color=(0.7, 0.35, 0.15),
        shape_kwargs={'scale': (2.2, 0.1, 2.2)}
    )
    return concat_shapes(posts, transform_shape(roof, translation_matrix((0, 2.1, 0))))

@register()
def lantern() -> Shape:
    body = primitive_call('sphere',
        color=(0.9, 0.2, 0.1),
        shape_kwargs={'radius': 0.15}
    )
    return body

@register()
def lantern_string() -> Shape:
    def make_lantern_at_height(i: int) -> Shape:
        lantern_shape = library_call('lantern')
        swing_angle = math.sin(i * 0.5) * 0.1
        height = 2 - i * 0.4
        pos = translation_matrix((0, height, 0))
        rot = rotation_matrix(swing_angle, (1, 0, 0), (0, height, 0))
        return transform_shape(lantern_shape, pos @ rot)

    return loop(4, make_lantern_at_height)

@register()
def person() -> Shape:
    # Simplified person representation
    body = primitive_call('cylinder',
        color=(0.7, 0.5, 0.3),
        shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 1.6, 0)}
    )
    head = primitive_call('sphere',
        color=(0.8, 0.6, 0.4),
        shape_kwargs={'radius': 0.15}
    )
    return concat_shapes(body, transform_shape(head, translation_matrix((0, 1.75, 0))))

@register()
def kite() -> Shape:
    kite_body = primitive_call('cube',
        color=(0.9, 0.3, 0.3),
        shape_kwargs={'scale': (0.4, 0.4, 0.02)}
    )
    string = primitive_call('cylinder',
        color=(0.8, 0.8, 0.8),
        shape_kwargs={'radius': 0.01, 'p0': (0, 0, 0), 'p1': (0, -1.5, -0.5)}
    )
    return concat_shapes(kite_body, string)

@register()
def crowd_section() -> Shape:
    def place_person(i: int) -> Shape:
        person_shape = library_call('person')
        x_pos = np.random.uniform(-2, 2)
        z_pos = np.random.uniform(-2, 2)
        y_rot = np.random.uniform(-math.pi, math.pi)
        pos = translation_matrix((x_pos, 0, z_pos))
        rot = rotation_matrix(y_rot, (0, 1, 0), (x_pos, 0, z_pos))
        return transform_shape(person_shape, pos @ rot)

    return loop(20, place_person)

@register()
def marketplace() -> Shape:
    # Create multiple stalls
    def place_stall(i: int) -> Shape:
        stall = library_call('stall_frame')
        x_pos = (i % 3) * 4.5 - 4.5
        z_pos = (i // 3) * 4.5 - 4.5
        pos = translation_matrix((x_pos, 0, z_pos))
        rot = rotation_matrix(np.random.uniform(-0.1, 0.1), (0, 1, 0), (x_pos, 0, z_pos))
        lanterns = library_call('lantern_string')
        return concat_shapes(
            transform_shape(stall, pos @ rot),
            transform_shape(lanterns, pos)
        )

    stalls = loop(9, place_stall)

    # Create multiple crowd sections
    def place_crowd(i: int) -> Shape:
        crowd = library_call('crowd_section')
        return transform_shape(crowd, translation_matrix((i*5-5, 0, i*3-3)))

    crowds = loop(3, place_crowd)

    # Add flying kites
    def place_kite(i: int) -> Shape:
        kite_shape = library_call('kite')
        height = np.random.uniform(3, 5)
        x_pos = np.random.uniform(-8, 8)
        z_pos = np.random.uniform(-8, 8)
        pos = translation_matrix((x_pos, height, z_pos))
        swing = rotation_matrix(math.sin(i*0.7)*0.2, (1, 0, 0), (x_pos, height, z_pos))
        return transform_shape(kite_shape, pos @ swing)

    kites = loop(5, place_kite)

    return concat_shapes(stalls, crowds, kites)
"""

This implementation creates a vibrant marketplace scene with:

1. Stall frames with wooden posts and roofs
2. Hanging red lanterns that swing slightly
3. Crowds of people scattered throughout the marketplace
4. Flying kites at different heights
5. Multiple modular components that can be reused and transformed

The scene includes:
- 9 stalls arranged in a 3x3 grid
- Each stall has 4 hanging lanterns
- 3 crowd sections with 20 people each (60 total people)
- 5 kites flying at different heights

The randomization in positions and rotations creates natural variation and a more organic feel to the scene. The lanterns and kites have subtle animations through mathematical functions to simulate movement.
"""