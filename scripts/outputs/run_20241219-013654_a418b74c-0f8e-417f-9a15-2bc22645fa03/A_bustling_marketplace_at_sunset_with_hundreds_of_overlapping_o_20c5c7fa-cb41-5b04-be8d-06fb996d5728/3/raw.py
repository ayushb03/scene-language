"""
I'll create a bustling marketplace scene with various components. I'll break it down into modular functions for different elements:

"""
from helper import *

"""
A bustling marketplace at sunset with hundreds of overlapping objects like food stalls,
hanging lanterns, moving crowds and children playing with kites while shadows lengthen dynamically
"""

@register()
def stall_frame(width: float, height: float, depth: float) -> Shape:
    """Creates a basic food stall frame"""
    posts = []
    for x, z in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        post = primitive_call('cylinder',
                            color=(0.4, 0.2, 0.1),
                            shape_kwargs={
                                'radius': 0.05,
                                'p0': (x * width/2, 0, z * depth/2),
                                'p1': (x * width/2, height, z * depth/2)
                            })
        posts.append(post)

    roof = primitive_call('cube',
                         color=(0.5, 0.3, 0.1),
                         shape_kwargs={'scale': (width*1.2, 0.1, depth*1.2)})
    return concat_shapes(*posts, transform_shape(roof, translation_matrix((0, height, 0))))

@register()
def lantern(radius: float) -> Shape:
    """Creates a paper lantern"""
    sphere = primitive_call('sphere',
                          color=(0.9, 0.6, 0.2),
                          shape_kwargs={'radius': radius})
    wire = primitive_call('cylinder',
                         color=(0.3, 0.3, 0.3),
                         shape_kwargs={
                             'radius': radius/10,
                             'p0': (0, radius, 0),
                             'p1': (0, radius*2, 0)
                         })
    return concat_shapes(sphere, wire)

@register()
def kite() -> Shape:
    """Creates a diamond-shaped kite"""
    kite_body = primitive_call('cube',
                             color=(0.8, 0.2, 0.2),
                             shape_kwargs={'scale': (0.3, 0.3, 0.02)})
    string = primitive_call('cylinder',
                          color=(0.8, 0.8, 0.8),
                          shape_kwargs={
                              'radius': 0.005,
                              'p0': (0, 0, 0),
                              'p1': (0, -1, 0)
                          })
    return concat_shapes(transform_shape(kite_body, rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0))),
                        string)

@register()
def person(height: float) -> Shape:
    """Creates a simple person representation"""
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
    return concat_shapes(body, transform_shape(head, translation_matrix((0, height*0.7, 0))))

@register()
def marketplace_scene() -> Shape:
    scene = []

    # Create multiple stalls
    def stall_loop(i: int) -> Shape:
        x_pos = (i % 3 - 1) * 3
        z_pos = (i // 3 - 1) * 3
        stall = library_call('stall_frame', width=2, height=2.5, depth=2)
        return transform_shape(stall, translation_matrix((x_pos, 0, z_pos)))

    stalls = loop(9, stall_loop)
    scene.append(stalls)

    # Add hanging lanterns
    def lantern_loop(i: int) -> Shape:
        x_pos = np.random.uniform(-4, 4)
        z_pos = np.random.uniform(-4, 4)
        y_pos = np.random.uniform(2, 3)
        lantern_shape = library_call('lantern', radius=0.2)
        return transform_shape(lantern_shape, translation_matrix((x_pos, y_pos, z_pos)))

    lanterns = loop(20, lantern_loop)
    scene.append(lanterns)

    # Add people
    def person_loop(i: int) -> Shape:
        x_pos = np.random.uniform(-5, 5)
        z_pos = np.random.uniform(-5, 5)
        rotation = np.random.uniform(0, 2*math.pi)
        person_shape = library_call('person', height=1.7)
        person_shape = transform_shape(person_shape, translation_matrix((x_pos, 0, z_pos)))
        return transform_shape(person_shape,
                             rotation_matrix(rotation, (0, 1, 0), compute_shape_center(person_shape)))

    people = loop(30, person_loop)
    scene.append(people)

    # Add flying kites
    def kite_loop(i: int) -> Shape:
        x_pos = np.random.uniform(-6, 6)
        z_pos = np.random.uniform(-6, 6)
        y_pos = np.random.uniform(3, 5)
        kite_shape = library_call('kite')
        return transform_shape(kite_shape, translation_matrix((x_pos, y_pos, z_pos)))

    kites = loop(5, kite_loop)
    scene.append(kites)

    return concat_shapes(*scene)
"""

This program creates a lively marketplace scene with:
1. A 3x3 grid of food stalls with wooden frames and roofs
2. Multiple glowing lanterns hanging at different heights
3. A crowd of people scattered throughout the marketplace
4. Children's kites flying above the scene

Each component is modular and can be easily modified. The scene uses randomization to create natural variation in the positioning of people, lanterns, and kites. The warm colors (especially in the lanterns) suggest a sunset atmosphere.

The marketplace is organized with:
- Stalls arranged in a grid pattern for clear pathways
- Lanterns providing ambient lighting
- Crowds of people distributed throughout the space
- Kites flying high above to add vertical interest

The scene creates depth through overlapping elements and varying heights, from ground-level people to high-flying kites.
"""