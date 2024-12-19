"""
I'll create an artistic representation of an ancient tapestry using geometric shapes and patterns. I'll break it down into components: the tapestry frame, the background fabric, and decorative patterns.

"""
from helper import *

"""
An ancient tapestry in a dimly lit museum, with intricate woven patterns depicting mythical creatures, each thread reflecting a different color when illuminated by flickering candles.
"""

@register()
def wooden_frame(width: float, height: float, thickness: float) -> Shape:
    """Creates an ornate wooden frame for the tapestry"""
    frame_color = (0.4, 0.2, 0.1)  # Dark wood color

    # Create frame pieces
    top = primitive_call('cube', shape_kwargs={'scale': (width, thickness, thickness)}, color=frame_color)
    bottom = primitive_call('cube', shape_kwargs={'scale': (width, thickness, thickness)}, color=frame_color)
    left = primitive_call('cube', shape_kwargs={'scale': (thickness, height, thickness)}, color=frame_color)
    right = primitive_call('cube', shape_kwargs={'scale': (thickness, height, thickness)}, color=frame_color)

    # Position frame pieces
    bottom = transform_shape(bottom, translation_matrix((0, -height/2, 0)))
    top = transform_shape(top, translation_matrix((0, height/2, 0)))
    left = transform_shape(left, translation_matrix((-width/2, 0, 0)))
    right = transform_shape(right, translation_matrix((width/2, 0, 0)))

    return concat_shapes(top, bottom, left, right)

@register()
def fabric_background(width: float, height: float) -> Shape:
    """Creates the base fabric with subtle texture"""
    def texture_element(i: int) -> Shape:
        x = (i % 10) * (width/10) - width/2 + width/20
        y = (i // 10) * (height/10) - height/2 + height/20

        # Vary colors slightly for texture
        color = (0.6 + np.random.uniform(-0.1, 0.1),
                0.5 + np.random.uniform(-0.1, 0.1),
                0.4 + np.random.uniform(-0.1, 0.1))

        square = primitive_call('cube',
                              shape_kwargs={'scale': (width/10, height/10, 0.01)},
                              color=color)
        return transform_shape(square, translation_matrix((x, y, 0.01)))

    return loop(100, texture_element)  # 10x10 grid of slightly varying colored squares

@register()
def mythical_pattern(base_x: float, base_y: float) -> Shape:
    """Creates a decorative pattern suggesting a mythical creature"""
    patterns = []

    # Create spiral pattern suggesting dragon tail
    def spiral_element(i: int) -> Shape:
        angle = i * 0.5
        radius = 0.05 - (i * 0.002)
        x = base_x + radius * math.cos(angle)
        y = base_y + radius * math.sin(angle)

        color = (0.8 + np.random.uniform(-0.1, 0.1),
                0.6 + np.random.uniform(-0.1, 0.1),
                0.2 + np.random.uniform(-0.1, 0.1))

        sphere = primitive_call('sphere',
                              shape_kwargs={'radius': 0.01},
                              color=color)
        return transform_shape(sphere, translation_matrix((x, y, 0.02)))

    patterns.extend(loop(20, spiral_element))

    # Add central body
    body = primitive_call('sphere',
                         shape_kwargs={'radius': 0.03},
                         color=(0.9, 0.7, 0.2))
    body = transform_shape(body, translation_matrix((base_x, base_y, 0.02)))
    patterns.append(body)

    return concat_shapes(*patterns)

@register()
def ancient_tapestry() -> Shape:
    width, height = 1.0, 1.5  # Tapestry dimensions

    # Create base components
    frame = library_call('wooden_frame', width=width+0.1, height=height+0.1, thickness=0.05)
    background = library_call('fabric_background', width=width, height=height)

    # Create multiple mythical patterns
    patterns = []
    pattern_positions = [
        (-0.2, 0.3), (0.2, -0.2), (0.0, 0.0),
        (-0.3, -0.4), (0.3, 0.4)
    ]

    for x, y in pattern_positions:
        pattern = library_call('mythical_pattern', base_x=x, base_y=y)
        patterns.append(pattern)

    # Combine all elements
    return concat_shapes(
        frame,
        background,
        *patterns
    )
"""

This program creates:
1. A wooden frame using four rectangular prisms arranged in a border
2. A textured background made of slightly varying colored squares
3. Multiple mythical patterns, each consisting of:
   - A spiral pattern suggesting a dragon's tail (made of small spheres with varying colors)
   - A central body (larger sphere)
4. All elements are combined with proper depth ordering (background behind patterns, frame in front)

The result suggests an ancient tapestry with:
- Rich wooden frame
- Textured fabric background
- Multiple golden-hued patterns suggesting mythical creatures
- Subtle color variations simulating the effect of flickering candlelight
- Depth and dimensionality through layering of elements

The patterns are abstract enough to suggest mythical creatures while maintaining a geometric simplicity that works well with the available primitive shapes.
"""