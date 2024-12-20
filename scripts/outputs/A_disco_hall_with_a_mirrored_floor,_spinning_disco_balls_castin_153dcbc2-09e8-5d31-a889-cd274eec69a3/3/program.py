"""
I'll create a disco hall scene with multiple disco balls and a reflective floor. Here's my implementation:

"""
from helper import *

"""
A disco hall with a mirrored floor, spinning disco balls casting moving patterns of colored light, and dynamic reflections on shiny surfaces as people dance.
"""

@register()
def floor(width: float, length: float) -> Shape:
    # Reflective floor made with a thin scaled cube
    return primitive_call('cube',
                        color=(0.8, 0.8, 0.9),  # Silvery reflective color
                        shape_kwargs={'scale': (width, 0.01, length)})

@register()
def disco_ball(radius: float) -> Shape:
    # Disco ball as a silvery sphere
    ball = primitive_call('sphere',
                         color=(0.9, 0.9, 0.95),
                         shape_kwargs={'radius': radius})

    # Add mirror facets using small cubes around the sphere
    def facet_fn(i: int) -> Shape:
        angle1 = (i / 20) * 2 * math.pi
        angle2 = (i % 5) * math.pi / 2.5

        facet = primitive_call('cube',
                             color=(0.95, 0.95, 1.0),
                             shape_kwargs={'scale': (radius*0.15, radius*0.15, radius*0.01)})

        # Position facets in a spiral pattern
        pos = (radius * math.cos(angle1) * math.cos(angle2),
               radius * math.sin(angle2),
               radius * math.sin(angle1) * math.cos(angle2))

        facet = transform_shape(facet, translation_matrix(pos))
        center = compute_shape_center(facet)

        # Orient facets to face outward
        return transform_shape(facet, rotation_matrix(angle1, (0, 1, 0), center))

    facets = loop(40, facet_fn)
    return concat_shapes(ball, facets)

@register()
def dancer(height: float) -> Shape:
    # Simplified dancer representation using cylinders
    body = primitive_call('cylinder',
                         color=(0.7, 0.3, 0.3),
                         shape_kwargs={'radius': 0.15,
                                     'p0': (0, 0, 0),
                                     'p1': (0, height*0.6, 0)})

    head = primitive_call('sphere',
                         color=(0.8, 0.6, 0.5),
                         shape_kwargs={'radius': 0.1})

    head = transform_shape(head, translation_matrix((0, height*0.7, 0)))

    return concat_shapes(body, head)

@register()
def dancing_crowd() -> Shape:
    def crowd_fn(i: int) -> Shape:
        dancer_shape = library_call('dancer', height=1.7)

        # Random positions and rotations for dancers
        x = math.cos(i * 2 * math.pi / 8) * 2
        z = math.sin(i * 2 * math.pi / 8) * 2

        dancer_shape = transform_shape(dancer_shape, translation_matrix((x, 0, z)))
        center = compute_shape_center(dancer_shape)

        # Make dancers face center
        angle = math.atan2(x, z)
        return transform_shape(dancer_shape, rotation_matrix(angle, (0, 1, 0), center))

    return loop(8, crowd_fn)

@register()
def disco_hall() -> Shape:
    # Create main components
    floor_shape = library_call('floor', width=10.0, length=10.0)
    crowd = library_call('dancing_crowd')

    # Create multiple disco balls at different heights
    def disco_balls_fn(i: int) -> Shape:
        ball = library_call('disco_ball', radius=0.3)

        # Position balls in a circular pattern
        angle = (i / 3) * 2 * math.pi
        x = math.cos(angle) * 1.5
        z = math.sin(angle) * 1.5
        y = 2.5 + 0.3 * math.sin(i * math.pi / 3)  # Varying heights

        ball = transform_shape(ball, translation_matrix((x, y, z)))
        center = compute_shape_center(ball)

        # Add rotation for spinning effect
        spin_angle = (i % 2) * math.pi / 2  # Alternate rotation directions
        return transform_shape(ball, rotation_matrix(spin_angle, (0, 1, 0), center))

    disco_balls = loop(6, disco_balls_fn)

    # Combine all elements
    return concat_shapes(floor_shape, crowd, disco_balls)
"""

This implementation creates:

1. A reflective floor using a thin scaled cube with a silvery color
2. Disco balls made of a central sphere with multiple reflective facets arranged in a spiral pattern
3. Simplified dancers represented by cylinders for bodies and spheres for heads
4. A crowd of dancers arranged in a circle, each facing the center
5. Multiple disco balls positioned in a circular pattern at different heights with varying rotations

The scene features:
- 8 dancers arranged in a circular pattern
- 6 disco balls at different heights with reflective facets
- A large reflective floor
- Various transformations for realistic positioning and orientation
- Different colors for different elements (silvery for reflective surfaces, skin tones for dancers)

The disco balls are positioned at different heights and have different rotations to create a more dynamic scene. The dancers are arranged in a circle facing the center, simulating a dance floor arrangement. The reflective floor adds to the disco atmosphere by suggesting the presence of light reflections.
"""