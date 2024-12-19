"""
I'll create a coral reef scene with branching corals, fish schools, and decorative elements. Here's my implementation:

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
        color=(0.3, 0.5, 0.9))
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
def seaweed(height: float) -> Shape:
    """Creates a swaying piece of seaweed"""
    def weed_segment(i: int) -> Shape:
        segment = primitive_call('cylinder',
            shape_kwargs={'radius': 0.02,
                        'p0': (0, i * height/8, 0),
                        'p1': (0, (i+1) * height/8, 0)},
            color=(0.2, 0.8, 0.3))
        angle = math.sin(i * 0.8) * 0.3
        return transform_shape(segment,
            rotation_matrix(angle, (1, 0, 0), (0, i * height/8, 0)))

    return loop(8, weed_segment)

@register()
def coral_reef() -> Shape:
    """Assembles the complete coral reef scene"""
    # Create base rock
    base = primitive_call('sphere',
        shape_kwargs={'radius': 1.2},
        color=(0.6, 0.6, 0.6))
    base = transform_shape(base, scale_matrix(0.5, (0, 0, 0)))

    # Add coral clusters
    corals = library_call('coral_cluster')

    # Add multiple schools of fish at different heights
    fish_school1 = library_call('fish_school')
    fish_school2 = transform_shape(
        library_call('fish_school'),
        translation_matrix((0, 0.8, 0)))

    # Add seaweed patches
    def place_seaweed(i: int) -> Shape:
        angle = i * math.pi * 2 / 6
        radius = 0.8
        weed = library_call('seaweed', height=0.8)
        pos = (radius * math.cos(angle), 0, radius * math.sin(angle))
        return transform_shape(weed, translation_matrix(pos))

    seaweed_patch = loop(6, place_seaweed)

    return concat_shapes(
        base,
        corals,
        fish_school1,
        fish_school2,
        seaweed_patch
    )
"""

This implementation creates a vibrant coral reef scene with:

1. Branching coral structures that curve organically
2. Multiple schools of fish swimming in circular patterns at different heights
3. Swaying seaweed patches around the edges
4. A base rock structure
5. Natural variation through random offsets and sinusoidal movements

The scene is built modularly, with each component (coral_branch, fish, seaweed) defined separately and then combined into larger structures (coral_cluster, fish_school). The final coral_reef function assembles everything together.

The corals have a pink-orange color gradient, the fish are blue, and the seaweed is green, creating a colorful underwater scene. The fish schools swim in circular patterns at different heights, suggesting movement, while the seaweed sways as if moved by ocean currents.
"""