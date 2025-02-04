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
def hanging_wire(height: float) -> Shape:
    # Thin wire to hang disco ball
    return primitive_call('cylinder',
                        color=(0.3, 0.3, 0.3),
                        shape_kwargs={
                            'radius': 0.01,
                            'p0': (0, 0, 0),
                            'p1': (0, height, 0)
                        })

@register()
def spotlight(radius: float, color: tuple[float, float, float]) -> Shape:
    # Colored spotlight as a cylinder
    return primitive_call('cylinder',
                        color=color,
                        shape_kwargs={
                            'radius': radius,
                            'p0': (0, 0, 0),
                            'p1': (0, -2, 0)  # Points downward
                        })

@register()
def disco_ball_assembly(position: P, ball_radius: float, wire_height: float) -> Shape:
    # Create a disco ball with hanging wire and rotating spotlights
    ball = library_call('disco_ball', radius=ball_radius)
    wire = library_call('hanging_wire', height=wire_height)

    def spotlight_fn(i: int) -> Shape:
        # Create multiple spotlights around the disco ball
        spot = library_call('spotlight', radius=0.1,
                          color=(np.random.uniform(0.5, 1),
                                np.random.uniform(0.5, 1),
                                np.random.uniform(0.5, 1)))
        angle = (i / 6) * 2 * math.pi  # Evenly space spotlights
        # Rotate spotlight around ball and point outward
        transform = rotation_matrix(angle, (0, 1, 0), (0, 0, 0))
        return transform_shape(spot, transform)

    spotlights = loop(6, spotlight_fn)  # Create 6 spotlights

    # Combine all elements and position them
    assembly = concat_shapes(ball, wire, spotlights)
    return transform_shape(assembly, translation_matrix(position))

@register()
def disco_hall() -> Shape:
    # Create the main disco hall scene
    hall_width = 10
    hall_length = 10

    # Create reflective floor
    floor_shape = library_call('floor', width=hall_width, length=hall_length)

    # Create multiple disco balls at different positions
    disco_balls = concat_shapes(
        library_call('disco_ball_assembly',
                    position=(-2, 3, -2),
                    ball_radius=0.3,
                    wire_height=1),
        library_call('disco_ball_assembly',
                    position=(2, 4, 2),
                    ball_radius=0.4,
                    wire_height=0.5),
        library_call('disco_ball_assembly',
                    position=(0, 3.5, 0),
                    ball_radius=0.5,
                    wire_height=0.7)
    )

    # Add random rotation to simulate spinning disco balls
    disco_center = compute_shape_center(disco_balls)
    spinning_balls = transform_shape(
        disco_balls,
        rotation_matrix(np.random.uniform(0, 2*math.pi), (0, 1, 0), disco_center)
    )

    # Combine all elements
    return concat_shapes(floor_shape, spinning_balls)
"""

This implementation creates:

1. A reflective floor using a thin scaled cube with a silvery color
2. Disco balls represented as metallic spheres
3. Hanging wires for the disco balls using thin cylinders
4. Colored spotlights around each disco ball using cylinders
5. Multiple disco ball assemblies at different positions in the hall
6. Random rotations to simulate spinning motion
7. A modular design where each component is a separate function

The scene features:
- Multiple disco balls at different heights and positions
- Each disco ball has 6 colored spotlights rotating around it
- A large reflective floor
- Randomized colors for the spotlights to create dynamic lighting effects
- Spinning effect through random rotations
- Proper scaling and positioning of all elements

The scene creates a dynamic disco atmosphere with multiple light sources and reflective surfaces, capturing the essence of a disco hall.
"""