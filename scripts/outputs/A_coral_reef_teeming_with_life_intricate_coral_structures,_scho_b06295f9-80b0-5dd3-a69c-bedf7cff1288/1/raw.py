"""
I'll create a coral reef scene with branching corals, schools of fish, and other marine elements. Here's my implementation:

"""
from helper import *

"""
A coral reef teeming with life: intricate coral structures, schools of fish swimming in intricate patterns, and sunlight refracted through moving ocean waves.
"""

@register()
def coral_branch(height: float, radius: float) -> Shape:
    """Creates a single coral branch with organic curves"""
    def branch_segment(i: int) -> Shape:
        angle = math.sin(i * 0.5) * 0.3
        segment = primitive_call('cylinder',
            shape_kwargs={'radius': radius * (1 - i/10),
                        'p0': (0, i * height/10, 0),
                        'p1': (0, (i+1) * height/10, 0)},
            color=(1.0, 0.4 + i/20, 0.4))
        return transform_shape(segment,
                rotation_matrix(angle, (1, 0, 1), (0, i * height/10, 0)))

    return loop(10, branch_segment)

@register()
def coral_cluster() -> Shape:
    """Creates a cluster of coral branches"""
    def place_branch(i: int) -> Shape:
        angle = i * math.pi * 2 / 8
        radius = 0.3 + np.random.uniform(-0.1, 0.1)
        height = 1.0 + np.random.uniform(-0.2, 0.2)
        branch = library_call('coral_branch', height=height, radius=0.05)
        pos = (radius * math.cos(angle), 0, radius * math.sin(angle))
        return transform_shape(branch, translation_matrix(pos))

    return loop(8, place_branch)

@register()
def fish(size: float) -> Shape:
    """Creates a simple fish shape using spheres and cones"""
    body = primitive_call('sphere',
        shape_kwargs={'radius': size},
        color=(0.4, 0.6, 1.0))
    tail = primitive_call('cylinder',
        shape_kwargs={'radius': size*0.5,
                     'p0': (0, 0, size),
                     'p1': (0, size, size*2)},
        color=(0.4, 0.6, 1.0))
    return concat_shapes(body, tail)

@register()
def fish_school() -> Shape:
    """Creates a school of fish in a circular pattern"""
    def place_fish(i: int) -> Shape:
        angle = i * math.pi * 2 / 12
        radius = 1.0
        fish_shape = library_call('fish', size=0.1)
        pos = (radius * math.cos(angle), 0.5 + math.sin(i * 0.5) * 0.2, radius * math.sin(angle))
        rotated = transform_shape(fish_shape,
            rotation_matrix(angle + math.pi, (0, 1, 0), (0, 0, 0)))
        return transform_shape(rotated, translation_matrix(pos))

    return loop(12, place_fish)

@register()
def rock_formation() -> Shape:
    """Creates a base rock formation for the reef"""
    def place_rock(i: int) -> Shape:
        size = 0.3 + np.random.uniform(-0.1, 0.1)
        rock = primitive_call('sphere',
            shape_kwargs={'radius': size},
            color=(0.5, 0.5, 0.5))
        pos = (np.random.uniform(-1, 1), 0, np.random.uniform(-1, 1))
        return transform_shape(rock, translation_matrix(pos))

    return loop(6, place_rock)

@register()
def coral_reef_scene() -> Shape:
    """Assembles the complete coral reef scene"""
    # Create base rocks
    base = library_call('rock_formation')

    # Add multiple coral clusters
    def place_coral_cluster(i: int) -> Shape:
        cluster = library_call('coral_cluster')
        pos = (np.random.uniform(-0.8, 0.8), 0, np.random.uniform(-0.8, 0.8))
        return transform_shape(cluster, translation_matrix(pos))

    corals = loop(3, place_coral_cluster)

    # Add schools of fish at different heights
    def place_fish_school(i: int) -> Shape:
        school = library_call('fish_school')
        pos = (0, 0.5 + i * 0.5, 0)
        return transform_shape(school, translation_matrix(pos))

    fish_schools = loop(3, place_fish_school)

    return concat_shapes(base, corals, fish_schools)
"""

This implementation creates a vibrant coral reef scene with:

1. `coral_branch`: Creates organic-looking coral branches using cylinders with varying radii and slight curves
2. `coral_cluster`: Arranges multiple coral branches in a circular pattern
3. `fish`: Creates simple fish shapes using spheres and cylinders
4. `fish_school`: Arranges fish in circular swimming patterns
5. `rock_formation`: Creates the base rock structure for the reef
6. `coral_reef_scene`: Combines all elements into a complete scene with:
   - Rock formations at the base
   - Multiple coral clusters distributed across the base
   - Multiple schools of fish at different heights

The scene uses various transformations to create organic-looking arrangements and movements. Colors are chosen to represent typical coral reef hues, with pinkish corals, blue fish, and grey rocks. The fish schools are arranged in circular patterns at different heights to create the illusion of movement and life in the scene.
"""