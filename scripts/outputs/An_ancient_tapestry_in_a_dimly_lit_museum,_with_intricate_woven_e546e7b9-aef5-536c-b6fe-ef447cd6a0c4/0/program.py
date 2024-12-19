"""
I'll create a program that generates an artistic representation of an ancient tapestry with mythical patterns. I'll break it down into modular components:

"""
from helper import *

"""
An ancient tapestry in a dimly lit museum, with intricate woven patterns depicting mythical creatures, each thread reflecting a different color when illuminated by flickering candles.
"""

@register()
def tapestry_frame(width: float, height: float, thickness: float) -> Shape:
    """Creates an ornate wooden frame for the tapestry"""
    # Create main frame pieces
    frame_color = (0.4, 0.2, 0.1)  # Dark wood color
    left = primitive_call('cube', shape_kwargs={'scale': (thickness, height, thickness)}, color=frame_color)
    right = primitive_call('cube', shape_kwargs={'scale': (thickness, height, thickness)}, color=frame_color)
    top = primitive_call('cube', shape_kwargs={'scale': (width + 2*thickness, thickness, thickness)}, color=frame_color)
    bottom = primitive_call('cube', shape_kwargs={'scale': (width + 2*thickness, thickness, thickness)}, color=frame_color)

    # Position frame pieces
    right = transform_shape(right, translation_matrix((width/2 + thickness/2, 0, 0)))
    left = transform_shape(left, translation_matrix((-width/2 - thickness/2, 0, 0)))
    top = transform_shape(top, translation_matrix((0, height/2 + thickness/2, 0)))
    bottom = transform_shape(bottom, translation_matrix((0, -height/2 - thickness/2, 0)))

    return concat_shapes(left, right, top, bottom)

@register()
def mythical_pattern(size: float, color: tuple[float, float, float]) -> Shape:
    """Creates a single mythical pattern element"""
    # Create a spiral-like pattern using cylinders
    def spiral_fn(i: int) -> Shape:
        radius = 0.02
        angle = i * math.pi / 8
        r = (size/4) * (1 - i/16)
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        return primitive_call('cylinder',
                            shape_kwargs={'radius': radius,
                                        'p0': (x, y, 0),
                                        'p1': (x, y, 0.01)},
                            color=color)

    return loop(16, spiral_fn)

@register()
def pattern_grid(width: float, height: float) -> Shape:
    """Creates a grid of mythical patterns"""
    def grid_fn(i: int) -> Shape:
        row = i // 4
        col = i % 4

        # Vary colors slightly for each pattern
        base_hue = (0.1 + row * 0.2) % 1.0
        color = (base_hue, 0.5 + col * 0.1, 0.5)

        pattern = library_call('mythical_pattern', size=0.3, color=color)
        x = (col - 1.5) * 0.35
        y = (row - 1) * 0.35

        return transform_shape(pattern, translation_matrix((x, y, 0)))

    return loop(12, grid_fn)

@register()
def tapestry_fabric(width: float, height: float) -> Shape:
    """Creates the main fabric of the tapestry"""
    # Base fabric
    base = primitive_call('cube',
                         shape_kwargs={'scale': (width, height, 0.02)},
                         color=(0.8, 0.7, 0.6))  # Aged fabric color

    # Add patterns
    patterns = library_call('pattern_grid', width=width, height=height)

    return concat_shapes(base, patterns)

@register()
def ancient_tapestry() -> Shape:
    """Creates the complete ancient tapestry with frame"""
    width = 2.0
    height = 1.5

    # Create main components
    frame = library_call('tapestry_frame', width=width, height=height, thickness=0.1)
    fabric = library_call('tapestry_fabric', width=width, height=height)

    # Offset fabric slightly behind frame
    fabric = transform_shape(fabric, translation_matrix((0, 0, 0.05)))

    return concat_shapes(frame, fabric)
"""

This program creates:
1. A wooden frame (`tapestry_frame`) with ornate borders
2. Individual mythical patterns (`mythical_pattern`) using spiral-like cylindrical elements
3. A grid of patterns (`pattern_grid`) with varying colors to create visual interest
4. The main tapestry fabric (`tapestry_fabric`) combining a base layer with patterns
5. The complete assembled tapestry (`ancient_tapestry`)

The design features:
- Varying colors that simulate thread reflections
- Spiral patterns suggesting mythical creatures
- An aged wooden frame
- A subtle depth effect with the fabric slightly recessed
- A grid layout of repeating but color-varied patterns

The colors and patterns are designed to suggest an ancient, mystical quality while maintaining visual interest through subtle variations in hue and placement.
"""