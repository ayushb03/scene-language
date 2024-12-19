"""
I'll create an artistic representation of an ancient tapestry using geometric shapes. I'll break it down into patterns, borders, and mythical creature silhouettes.

"""
from helper import *

"""
An ancient tapestry in a dimly lit museum, with intricate woven patterns depicting mythical creatures, each thread reflecting a different color when illuminated by flickering candles.
"""

@register()
def tapestry_base(width: float, height: float) -> Shape:
    """Creates the base fabric of the tapestry"""
    return primitive_call('cube',
                        shape_kwargs={'scale': (width, height, 0.02)},
                        color=(0.4, 0.2, 0.15))  # deep burgundy base

@register()
def dragon_silhouette(scale: float) -> Shape:
    """Creates a simplified dragon shape using primitive shapes"""
    body = primitive_call('cylinder',
                        shape_kwargs={'radius': 0.1 * scale,
                                    'p0': (0, 0, 0),
                                    'p1': (0.5 * scale, 0.2 * scale, 0)},
                        color=(0.7, 0.5, 0.1))
    head = primitive_call('sphere',
                         shape_kwargs={'radius': 0.15 * scale},
                         color=(0.7, 0.5, 0.1))
    return concat_shapes(body,
                        transform_shape(head,
                                      translation_matrix((0.6 * scale, 0.25 * scale, 0))))

@register()
def geometric_pattern(size: float) -> Shape:
    """Creates a geometric pattern element"""
    def pattern_element(i: int) -> Shape:
        angle = i * (2 * math.pi / 8)
        diamond = primitive_call('cube',
                               shape_kwargs={'scale': (size * 0.2, size * 0.2, 0.01)},
                               color=(0.6 + i * 0.05, 0.3, 0.2))
        center = (0, 0, 0)
        rotated = transform_shape(diamond,
                                rotation_matrix(angle, (0, 0, 1), center))
        return transform_shape(rotated,
                             translation_matrix((size * 0.3 * math.cos(angle),
                                              size * 0.3 * math.sin(angle),
                                              0)))

    return loop(8, pattern_element)

@register()
def border_element(length: float) -> Shape:
    """Creates a decorative border element"""
    return primitive_call('cube',
                        shape_kwargs={'scale': (length, length * 0.15, 0.015)},
                        color=(0.8, 0.6, 0.2))  # gold trim

@register()
def tapestry_border(width: float, height: float) -> Shape:
    """Creates the ornate border of the tapestry"""
    def create_side(i: int) -> Shape:
        element = library_call('border_element', length=0.2)
        if i % 2 == 0:  # horizontal borders
            num_elements = int(width / 0.2)
            offset_x = (i == 0) * height/2 - (i == 2) * height/2
            offset_y = width/2 - 0.1
            return transform_shape(element,
                                 translation_matrix((offset_x, offset_y, 0)))
        else:  # vertical borders
            num_elements = int(height / 0.2)
            offset_x = width/2 - 0.1
            offset_y = (i == 1) * width/2 - (i == 3) * width/2
            element = transform_shape(element,
                                    rotation_matrix(math.pi/2, (0, 0, 1), (0, 0, 0)))
            return transform_shape(element,
                                 translation_matrix((offset_x, offset_y, 0)))

    return loop(4, create_side)

@register()
def ancient_tapestry() -> Shape:
    """Assembles the complete tapestry with all its elements"""
    width, height = 2.0, 3.0

    # Create base
    base = library_call('tapestry_base', width=width, height=height)

    # Add geometric patterns
    def pattern_grid(i: int) -> Shape:
        row = i // 3
        col = i % 3
        pattern = library_call('geometric_pattern', size=0.3)
        return transform_shape(pattern,
                             translation_matrix((
                                 -width/3 + col * width/3,
                                 height/3 - row * height/3,
                                 0.02)))

    patterns = loop(9, pattern_grid)

    # Add dragon silhouettes
    dragon1 = transform_shape(
        library_call('dragon_silhouette', scale=0.8),
        translation_matrix((-0.3, 0.5, 0.02)))

    dragon2 = transform_shape(
        library_call('dragon_silhouette', scale=0.8),
        concat_shapes(
            reflection_matrix((0, 0, 0), (1, 0, 0)),
            translation_matrix((0.3, -0.5, 0.02))))

    # Add border
    border = library_call('tapestry_border', width=width, height=height)

    return concat_shapes(base, patterns, dragon1, dragon2, border)
"""

This program creates:
1. A tapestry base with a deep burgundy color
2. Geometric patterns arranged in a 3x3 grid, each made of 8 rotating diamond shapes
3. Two dragon silhouettes facing each other, created from simple cylinders and spheres
4. An ornate border made of repeating elements
5. All elements are layered with slight z-offset to prevent z-fighting

The design uses warm colors (burgundy, gold) to suggest candlelight illumination, and the geometric patterns create visual interest while the dragon silhouettes add the mythical element. The border frames the entire composition.
"""