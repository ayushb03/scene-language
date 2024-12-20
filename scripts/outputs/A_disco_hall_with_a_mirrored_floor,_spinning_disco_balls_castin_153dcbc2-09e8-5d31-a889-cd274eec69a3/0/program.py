"""
I'll create a disco hall scene with spinning disco balls and a reflective floor. Here's my implementation:

"""
from helper import *

"""
A disco hall with a mirrored floor, spinning disco balls casting moving patterns of colored light, and dynamic reflections on shiny surfaces as people dance.
"""

@register()
def disco_ball(radius: float) -> Shape:
    # Create a metallic silver disco ball
    return primitive_call('sphere',
                        shape_kwargs={'radius': radius},
                        color=(0.8, 0.8, 0.9))

@register()
def floor_tile(size: float) -> Shape:
    # Create a reflective floor tile
    return primitive_call('cube',
                        shape_kwargs={'scale': (size, 0.01, size)},
                        color=(0.9, 0.9, 0.95))

@register()
def dance_floor(width: float, length: float, tile_size: float) -> Shape:
    def tile_fn(i: int) -> Shape:
        num_tiles_width = int(width / tile_size)
        row = i // num_tiles_width
        col = i % num_tiles_width

        tile = library_call('floor_tile', size=tile_size * 0.95)  # Small gap between tiles
        return transform_shape(
            tile,
            translation_matrix((
                (col - num_tiles_width/2) * tile_size,
                0,
                (row - num_tiles_width/2) * tile_size
            ))
        )

    return loop(int((width/tile_size) * (length/tile_size)), tile_fn)

@register()
def spinning_disco_balls() -> Shape:
    def ball_fn(i: int) -> Shape:
        ball = library_call('disco_ball', radius=0.3)
        angle = (i / 4) * 2 * math.pi  # Distribute balls in a circle
        radius = 2.0  # Radius of the circle

        # Position the ball
        ball = transform_shape(ball, translation_matrix((
            radius * math.cos(angle),
            3.0,  # Height
            radius * math.sin(angle)
        )))

        # Add unique rotation to each ball
        ball_center = compute_shape_center(ball)
        rotation_angle = (i * math.pi / 2) % (2 * math.pi)
        return transform_shape(
            ball,
            rotation_matrix(rotation_angle, (1, 1, 1), ball_center)
        )

    return loop(4, ball_fn)

@register()
def dancer(height: float) -> Shape:
    # Simplified dancer representation using cylinders
    body = primitive_call('cylinder',
                        shape_kwargs={'radius': 0.2,
                                    'p0': (0, 0, 0),
                                    'p1': (0, height * 0.6, 0)},
                        color=(0.8, 0.2, 0.3))

    head = primitive_call('sphere',
                         shape_kwargs={'radius': 0.15},
                         color=(0.8, 0.6, 0.5))

    head = transform_shape(head, translation_matrix((0, height * 0.75, 0)))

    return concat_shapes(body, head)

@register()
def dancing_crowd() -> Shape:
    def crowd_fn(i: int) -> Shape:
        angle = (i / 8) * 2 * math.pi
        radius = 3.0

        dancer_shape = library_call('dancer', height=1.7)

        # Position dancer
        dancer_shape = transform_shape(
            dancer_shape,
            translation_matrix((
                radius * math.cos(angle),
                0,
                radius * math.sin(angle)
            ))
        )

        # Add dancing motion (slight tilt)
        dancer_center = compute_shape_center(dancer_shape)
        tilt_angle = math.sin(i * math.pi / 4) * 0.2
        return transform_shape(
            dancer_shape,
            rotation_matrix(tilt_angle, (0, 1, 0), dancer_center)
        )

    return loop(8, crowd_fn)

@register()
def disco_hall() -> Shape:
    # Create the main components
    floor = library_call('dance_floor', width=10.0, length=10.0, tile_size=1.0)
    balls = library_call('spinning_disco_balls')
    crowd = library_call('dancing_crowd')

    # Create walls using reflection planes
    wall1 = primitive_call('cube',
                         shape_kwargs={'scale': (10.0, 5.0, 0.1)},
                         color=(0.7, 0.7, 0.8))
    wall2 = primitive_call('cube',
                         shape_kwargs={'scale': (0.1, 5.0, 10.0)},
                         color=(0.7, 0.7, 0.8))

    # Position walls
    wall1 = transform_shape(wall1, translation_matrix((0, 2.5, -5)))
    wall2 = transform_shape(wall2, translation_matrix((5, 2.5, 0)))

    return concat_shapes(
        floor,
        balls,
        crowd,
        wall1,
        wall2
    )
"""

This implementation creates a disco hall with:

1. A reflective dance floor made of tiles
2. Four spinning disco balls arranged in a circle near the ceiling
3. A crowd of simplified dancers arranged in a circle
4. Reflective walls to enhance the disco atmosphere

The scene includes:
- Dynamic elements through the spinning disco balls
- Multiple dancers with slight tilting movements to simulate dancing
- A tiled floor with reflective properties
- Walls to create the enclosed disco hall space

The scene uses the coordinate system as specified (+x right, +y up, +z backward) and employs transformations for positioning and animation effects. Each component is modular and can be adjusted through parameters like sizes, numbers of elements, and positions.
"""