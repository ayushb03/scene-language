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
    # Disco ball as a metallic sphere
    return primitive_call('sphere',
                        color=(0.9, 0.9, 0.9),  # Metallic silver
                        shape_kwargs={'radius': radius})

@register()
def light_beam(length: float, radius: float, color: tuple[float, float, float]) -> Shape:
    # Light beam as a colored cylinder
    return primitive_call('cylinder',
                        color=color,
                        shape_kwargs={
                            'radius': radius,
                            'p0': (0, 0, 0),
                            'p1': (0, -length, 0)
                        })

@register()
def disco_ball_with_lights(ball_radius: float, beam_count: int) -> Shape:
    ball = library_call('disco_ball', radius=ball_radius)

    def create_light_beam(i: int) -> Shape:
        # Create colorful light beams rotating around the disco ball
        angle = (i / beam_count) * 2 * math.pi
        color = (
            math.sin(angle) * 0.5 + 0.5,
            math.cos(angle * 2) * 0.5 + 0.5,
            math.sin(angle * 3) * 0.5 + 0.5
        )
        beam = library_call('light_beam', length=5.0, radius=0.1, color=color)

        # Position and rotate the beam
        transform = identity_matrix()
        transform = transform @ rotation_matrix(angle, (0, 1, 0), (0, 0, 0))
        transform = transform @ rotation_matrix(math.pi/4, (1, 0, 0), (0, 0, 0))
        return transform_shape(beam, transform)

    light_beams = loop(beam_count, create_light_beam)
    return concat_shapes(ball, light_beams)

@register()
def dancer(height: float) -> Shape:
    # Simplified dancer representation using cylinders and a sphere
    body = primitive_call('cylinder',
                        color=(0.7, 0.3, 0.3),
                        shape_kwargs={
                            'radius': 0.2,
                            'p0': (0, 0, 0),
                            'p1': (0, height * 0.6, 0)
                        })

    head = primitive_call('sphere',
                         color=(0.8, 0.6, 0.6),
                         shape_kwargs={'radius': 0.15})

    head = transform_shape(head, translation_matrix((0, height * 0.75, 0)))

    return concat_shapes(body, head)

@register()
def disco_hall() -> Shape:
    # Create the main floor
    floor_shape = library_call('floor', width=20.0, length=20.0)

    # Create multiple disco balls at different positions
    disco_ball_1 = library_call('disco_ball_with_lights', ball_radius=0.5, beam_count=8)
    disco_ball_2 = library_call('disco_ball_with_lights', ball_radius=0.3, beam_count=6)

    # Position the disco balls
    disco_ball_1 = transform_shape(disco_ball_1, translation_matrix((0, 5, 0)))
    disco_ball_2 = transform_shape(disco_ball_2, translation_matrix((3, 4, 2)))

    # Create dancers
    def create_dancer(i: int) -> Shape:
        dancer_shape = library_call('dancer', height=1.8)
        angle = (i / 6) * 2 * math.pi
        radius = 3.0
        pos_x = math.cos(angle) * radius
        pos_z = math.sin(angle) * radius

        # Add some random rotation to make dancers face different directions
        transform = translation_matrix((pos_x, 0, pos_z))
        transform = transform @ rotation_matrix(angle + math.pi, (0, 1, 0), (pos_x, 0, pos_z))
        return transform_shape(dancer_shape, transform)

    dancers = loop(6, create_dancer)

    # Combine all elements
    return concat_shapes(
        floor_shape,
        disco_ball_1,
        disco_ball_2,
        dancers
    )
"""

This implementation creates a disco hall scene with:

1. A large reflective floor using a thin scaled cube
2. Multiple disco balls at different heights
3. Colorful light beams emanating from each disco ball, creating dynamic lighting patterns
4. Multiple dancers arranged in a circle around the center
5. Each disco ball has its own set of rotating light beams with different colors

The scene features:
- Dynamic elements through the arrangement of light beams at different angles
- Multiple disco balls of different sizes to create varied lighting effects
- A simplified representation of dancers using basic shapes
- A large reflective floor to simulate the mirror-like surface of a disco hall
- Color variations in the light beams to create the classic disco atmosphere

The dancers are arranged in a circle around the center, and the disco balls are positioned at different heights and locations to create an interesting and dynamic scene. The light beams use trigonometric functions to create smooth color transitions and are arranged radially around each disco ball.
"""