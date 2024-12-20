"""
I'll create a bustling marketplace scene with multiple modular components. Here's my implementation:

"""
from helper import *

"""
A bustling marketplace at sunset with hundreds of overlapping objects like food stalls,
hanging lanterns, moving crowds and children playing with kites while shadows lengthen dynamically
"""

@register()
def stall_frame(width: float, height: float, depth: float) -> Shape:
    """Basic wooden frame for a market stall"""
    posts = []
    for x, z in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        post = primitive_call('cylinder',
                            color=(0.6, 0.4, 0.2),
                            shape_kwargs={
                                'radius': 0.05,
                                'p0': (x * width/2, 0, z * depth/2),
                                'p1': (x * width/2, height, z * depth/2)
                            })
        posts.append(post)

    roof = primitive_call('cube',
                         color=(0.7, 0.3, 0.1),
                         shape_kwargs={'scale': (width*1.2, 0.1, depth*1.2)})
    return concat_shapes(*posts,
                        transform_shape(roof, translation_matrix((0, height, 0))))

@register()
def lantern(radius: float) -> Shape:
    """Traditional paper lantern"""
    body = primitive_call('sphere',
                         color=(0.9, 0.6, 0.2),
                         shape_kwargs={'radius': radius})
    light = primitive_call('sphere',
                          color=(1.0, 0.95, 0.7),
                          shape_kwargs={'radius': radius*0.6})
    return concat_shapes(body, light)

@register()
def person(height: float) -> Shape:
    """Simple person representation"""
    body = primitive_call('cylinder',
                         color=(0.7, 0.5, 0.3),
                         shape_kwargs={
                             'radius': height/8,
                             'p0': (0, 0, 0),
                             'p1': (0, height*0.6, 0)
                         })
    head = primitive_call('sphere',
                         color=(0.8, 0.6, 0.4),
                         shape_kwargs={'radius': height/8})
    return concat_shapes(body,
                        transform_shape(head, translation_matrix((0, height*0.7, 0))))

@register()
def kite() -> Shape:
    """Flying kite with string"""
    kite_body = primitive_call('cube',
                              color=(0.9, 0.3, 0.3),
                              shape_kwargs={'scale': (0.3, 0.3, 0.02)})
    string = primitive_call('cylinder',
                           color=(0.8, 0.8, 0.8),
                           shape_kwargs={
                               'radius': 0.01,
                               'p0': (0, 0, 0),
                               'p1': (0, -1.5, -0.5)
                           })
    return concat_shapes(kite_body, string)

@register()
def food_display() -> Shape:
    """Food items displayed on stall"""
    def make_food(i: int) -> Shape:
        colors = [(0.8, 0.4, 0.2), (0.6, 0.8, 0.2), (0.9, 0.8, 0.2)]
        food = primitive_call('sphere',
                            color=colors[i % len(colors)],
                            shape_kwargs={'radius': 0.1})
        offset = (np.random.uniform(-0.3, 0.3),
                 np.random.uniform(0, 0.1),
                 np.random.uniform(-0.2, 0.2))
        return transform_shape(food, translation_matrix(offset))

    return loop(15, make_food)

@register()
def lantern_string(num_lanterns: int) -> Shape:
    """String of hanging lanterns"""
    def place_lantern(i: int) -> Shape:
        lantern_shape = library_call('lantern', radius=0.15)
        offset = (i * 0.8 - (num_lanterns-1)*0.4,
                 np.sin(i*0.8)*0.1,
                 0)
        return transform_shape(lantern_shape, translation_matrix(offset))

    return loop(num_lanterns, place_lantern)

@register()
def crowd(num_people: int) -> Shape:
    """Group of people"""
    def place_person(i: int) -> Shape:
        height = np.random.uniform(1.5, 1.8)
        person_shape = library_call('person', height=height)
        offset = (np.random.uniform(-3, 3),
                 0,
                 np.random.uniform(-3, 3))
        rotation = np.random.uniform(0, 2*np.pi)
        transformed = transform_shape(person_shape, translation_matrix(offset))
        center = compute_shape_center(transformed)
        return transform_shape(transformed,
                             rotation_matrix(rotation, (0, 1, 0), center))

    return loop(num_people, place_person)

@register()
def kite_group() -> Shape:
    """Group of flying kites"""
    def place_kite(i: int) -> Shape:
        kite_shape = library_call('kite')
        offset = (np.random.uniform(-2, 2),
                 np.random.uniform(3, 5),
                 np.random.uniform(-2, 0))
        transformed = transform_shape(kite_shape, translation_matrix(offset))
        center = compute_shape_center(transformed)
        rotation = np.random.uniform(-0.3, 0.3)
        return transform_shape(transformed,
                             rotation_matrix(rotation, (1, 0, 1), center))

    return loop(5, place_kite)

@register()
def marketplace() -> Shape:
    """Complete marketplace scene"""
    # Create multiple stalls
    stalls = []
    for i in range(6):
        angle = i * np.pi/3
        radius = 4
        pos = (radius * np.cos(angle), 0, radius * np.sin(angle))
        stall = concat_shapes(
            library_call('stall_frame', width=2, height=2.5, depth=1.5),
            transform_shape(library_call('food_display'),
                          translation_matrix((0, 0.8, 0)))
        )
        transformed = transform_shape(stall, translation_matrix(pos))
        center = compute_shape_center(transformed)
        stalls.append(transform_shape(transformed,
                                    rotation_matrix(angle, (0, 1, 0), center)))

    # Add lantern strings between stalls
    lanterns = transform_shape(
        library_call('lantern_string', num_lanterns=8),
        translation_matrix((0, 2.3, 0))
    )

    # Add crowd and kites
    crowd_shape = library_call('crowd', num_people=30)
    kites = library_call('kite_group')

    return concat_shapes(
        *stalls,
        lanterns,
        crowd_shape,
        kites
    )
"""

This implementation creates a vibrant marketplace scene with:

1. Market stalls arranged in a circular pattern, each with:
   - Wooden frame structure
   - Colorful food displays
2. Hanging lanterns strung between stalls
3. A crowd of people scattered throughout the marketplace
4. Children's kites flying overhead

The scene uses randomization to create natural variation in:
- Food placement on stalls
- People's heights and positions
- Kite positions and orientations
- Lantern arrangements

The marketplace has depth and dimension through:
- Circular arrangement of stalls
- Multiple layers (ground level crowds, mid-level stalls and lanterns, high-level kites)
- Overlapping elements creating visual complexity

The sunset atmosphere is suggested through:
- Warm colors in the lanterns (orange and yellow glows)
- The hanging lanterns providing ambient lighting
- Multiple shadow-casting elements
"""