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
    # Create wooden frame structure
    posts = loop(4, lambda i: primitive_call('cylinder',
        color=(0.6, 0.3, 0.1),
        shape_kwargs={'radius': 0.05,
                     'p0': (((i%2)-0.5)*2, 0, ((i//2)-0.5)*2),
                     'p1': (((i%2)-0.5)*2, 2, ((i//2)-0.5)*2)}))

    roof = primitive_call('cube', color=(0.7, 0.35, 0.15),
                         shape_kwargs={'scale': (2.2, 0.1, 2.2)})
    return concat_shapes(posts, transform_shape(roof, translation_matrix((0, 2.1, 0))))

@register()
def lantern() -> Shape:
    body = primitive_call('sphere', color=(0.9, 0.6, 0.2),
                         shape_kwargs={'radius': 0.15})
    return body

@register()
def person(height: float) -> Shape:
    # Simple person representation
    body = primitive_call('cylinder', color=(0.6, 0.6, 0.6),
                         shape_kwargs={'radius': 0.15,
                                     'p0': (0, 0, 0),
                                     'p1': (0, height*0.6, 0)})
    head = primitive_call('sphere', color=(0.8, 0.6, 0.5),
                         shape_kwargs={'radius': 0.12})
    return concat_shapes(body,
                        transform_shape(head, translation_matrix((0, height*0.7, 0))))

@register()
def kite() -> Shape:
    # Diamond-shaped kite
    kite_body = primitive_call('cube', color=(0.9, 0.3, 0.3),
                              shape_kwargs={'scale': (0.3, 0.3, 0.01)})
    string = primitive_call('cylinder', color=(0.8, 0.8, 0.8),
                           shape_kwargs={'radius': 0.01,
                                       'p0': (0, 0, 0),
                                       'p1': (0, -1, 0)})
    return concat_shapes(transform_shape(kite_body,
                        rotation_matrix(math.pi/4, (0,0,1), (0,0,0))),
                        string)

@register()
def food_display() -> Shape:
    # Create random food items
    return loop(6, lambda i: primitive_call('sphere',
        color=(0.8+np.random.uniform(-0.2,0.2),
               0.4+np.random.uniform(-0.2,0.2),
               0.2+np.random.uniform(-0.2,0.2)),
        shape_kwargs={'radius': 0.1}))

@register()
def market_stall() -> Shape:
    frame = library_call('stall_frame')
    foods = transform_shape(library_call('food_display'),
                          translation_matrix((0, 0.8, 0)))
    lanterns = loop(2, lambda i: transform_shape(
        library_call('lantern'),
        translation_matrix(((-1)**i * 0.8, 1.8, 0))))

    return concat_shapes(frame, foods, lanterns)

@register()
def marketplace() -> Shape:
    # Create multiple stalls
    stalls = loop(9, lambda i: transform_shape(
        library_call('market_stall'),
        translation_matrix((((i%3)-1)*4.5, 0, ((i//3)-1)*4.5))))

    # Add people walking around
    people = loop(20, lambda i: transform_shape(
        library_call('person', height=1.6+np.random.uniform(-0.2,0.2)),
        translation_matrix((np.random.uniform(-10,10),
                          0,
                          np.random.uniform(-10,10)))))

    # Add children with kites
    kite_scenes = loop(5, lambda i: concat_shapes(
        transform_shape(library_call('person', height=1.2),
                       translation_matrix((np.random.uniform(-8,8),
                                        0,
                                        np.random.uniform(-8,8)))),
        transform_shape(library_call('kite'),
                       translation_matrix((np.random.uniform(-8,8),
                                        2.5,
                                        np.random.uniform(-8,8))))))

    return concat_shapes(stalls, people, kite_scenes)
"""

This implementation creates a lively marketplace scene with:

1. Market stalls with:
   - Wooden frame structure
   - Food displays
   - Hanging lanterns

2. People:
   - Random heights and positions
   - Simple cylindrical body and spherical head representation

3. Children with kites:
   - Diamond-shaped kites with strings
   - Smaller person representations

4. Scene composition:
   - 9 market stalls arranged in a 3x3 grid
   - 20 randomly positioned people
   - 5 children with kites scattered around

The scene uses warm colors (oranges and reds) to suggest a sunset atmosphere. The randomization in positions and sizes creates natural variation and a bustling feel. The overlapping is achieved by random positioning of elements within the same space.

To use this scene, simply call:
```python
marketplace_scene = library_call('marketplace')
```
"""