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
            color=(1.0, 0.4 + i*0.05, 0.4 + i*0.05),
            shape_kwargs={
                'radius': radius * (1 - i*0.08),
                'p0': (0, i * height/10, 0),
                'p1': (0, (i+1) * height/10, 0)
            })
        center = compute_shape_center(segment)
        return transform_shape(segment,
            rotation_matrix(angle, (1, 0, 1), center))

    return loop(10, branch_segment)

@register()
def coral_cluster() -> Shape:
    """Creates a cluster of coral branches"""
    def place_branch(i: int) -> Shape:
        branch = library_call('coral_branch',
            height=np.random.uniform(0.8, 1.2),
            radius=np.random.uniform(0.03, 0.06))
        angle = i * (2 * math.pi / 8)
        radius = np.random.uniform(0.1, 0.3)
        pos = (radius * math.cos(angle), 0, radius * math.sin(angle))
        return transform_shape(branch, translation_matrix(pos))

    return loop(8, place_branch)

@register()
def fish() -> Shape:
    """Creates a simple fish shape"""
    body = primitive_call('sphere',
        color=(0.6, 0.8, 1.0),
        shape_kwargs={'radius': 0.05})
    tail = primitive_call('cube',
        color=(0.6, 0.8, 1.0),
        shape_kwargs={'scale': (0.02, 0.06, 0.04)})

    tail = transform_shape(tail, translation_matrix((0, 0, 0.05)))
    return concat_shapes(body, tail)

@register()
def fish_school() -> Shape:
    """Creates a school of fish in a circular pattern"""
    def place_fish(i: int) -> Shape:
        fish_shape = library_call('fish')
        angle = i * (2 * math.pi / 12)
        radius = 0.4 + math.sin(i * 0.5) * 0.1
        height = math.cos(i * 0.8) * 0.2
        pos = (radius * math.cos(angle), height + 0.5, radius * math.sin(angle))

        # Rotate fish to face tangent to circle
        center = compute_shape_center(fish_shape)
        rotated = transform_shape(fish_shape,
            rotation_matrix(angle + math.pi/2, (0, 1, 0), center))

        return transform_shape(rotated, translation_matrix(pos))

    return loop(12, place_fish)

@register()
def seaweed() -> Shape:
    """Creates a swaying piece of seaweed"""
    def segment(i: int) -> Shape:
        return primitive_call('cylinder',
            color=(0.2, 0.8, 0.3),
            shape_kwargs={
                'radius': 0.02,
                'p0': (math.sin(i * 0.5) * 0.05, i * 0.1, 0),
                'p1': (math.sin((i + 1) * 0.5) * 0.05, (i + 1) * 0.1, 0)
            })

    return loop(8, segment)

@register()
def coral_reef_scene() -> Shape:
    """Assembles the complete coral reef scene"""
    # Create base with multiple coral clusters
    def place_cluster(i: int) -> Shape:
        cluster = library_call('coral_cluster')
        pos = (np.random.uniform(-1, 1), 0, np.random.uniform(-1, 1))
        return transform_shape(cluster, translation_matrix(pos))

    coral_base = loop(5, place_cluster)

    # Add multiple schools of fish
    def place_school(i: int) -> Shape:
        school = library_call('fish_school')
        pos = (np.random.uniform(-0.5, 0.5), 0.2 * i, np.random.uniform(-0.5, 0.5))
        return transform_shape(school, translation_matrix(pos))

    fish_schools = loop(3, place_school)

    # Add seaweed patches
    def place_seaweed(i: int) -> Shape:
        weed = library_call('seaweed')
        pos = (np.random.uniform(-1, 1), 0, np.random.uniform(-1, 1))
        return transform_shape(weed, translation_matrix(pos))

    seaweed_patches = loop(10, place_seaweed)

    # Combine all elements
    return concat_shapes(coral_base, fish_schools, seaweed_patches)
"""

This implementation creates a vibrant coral reef scene with:

1. `coral_branch`: Creates individual coral branches with organic curves using cylinders that get thinner towards the top
2. `coral_cluster`: Arranges multiple coral branches in a circular pattern
3. `fish`: Creates a simple fish shape using a sphere for the body and a cube for the tail
4. `fish_school`: Arranges fish in a circular pattern with slight vertical variation to create a natural swimming pattern
5. `seaweed`: Creates swaying seaweed using connected cylinders with sinusoidal movement
6. `coral_reef_scene`: Combines multiple coral clusters, schools of fish, and seaweed patches into a complete scene

The scene features:
- Multiple coral clusters randomly placed on the seafloor
- Several schools of fish swimming at different heights
- Seaweed patches scattered throughout the scene
- Organic shapes and movements created through mathematical functions
- Color variations to create visual interest

The scene uses the coordinate system as specified (+x right, +y up, +z backward) and makes extensive use of transformation matrices for positioning and rotation of elements.
"""