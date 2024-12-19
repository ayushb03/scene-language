"""
I'll help create a futuristic cityscape with multiple components. Here's my implementation:

"""
from helper import *

"""
A sprawling futuristic cityscape at night, with thousands of illuminated buildings,
flying vehicles, and interconnected bridges, all dynamically lit by neon lights.
"""

@register()
def skyscraper(width: float, height: float, depth: float) -> Shape:
    # Create main building structure
    building = primitive_call('cube',
                            shape_kwargs={'scale': (width, height, depth)},
                            color=(0.2, 0.2, 0.3))  # Dark blue-gray base

    # Add glowing windows
    def window_loop(i):
        window_size = min(width, depth) * 0.15
        window = primitive_call('cube',
                              shape_kwargs={'scale': (window_size, window_size, 0.01)},
                              color=(1, 1, 0.8))  # Warm light

        x_pos = np.random.uniform(-width/2 + window_size, width/2 - window_size)
        y_pos = np.random.uniform(-height/2 + window_size, height/2 - window_size)
        window = transform_shape(window, translation_matrix((x_pos, y_pos, depth/2)))
        return window

    windows = loop(int(height * width * 2), window_loop)
    return concat_shapes(building, windows)

@register()
def flying_vehicle() -> Shape:
    # Create vehicle body
    body = primitive_call('cube',
                         shape_kwargs={'scale': (0.3, 0.1, 0.5)},
                         color=(0.7, 0.7, 0.8))

    # Add glowing engines
    engine1 = primitive_call('sphere',
                           shape_kwargs={'radius': 0.05},
                           color=(1, 0.3, 0))
    engine2 = primitive_call('sphere',
                           shape_kwargs={'radius': 0.05},
                           color=(1, 0.3, 0))

    engines = concat_shapes(
        transform_shape(engine1, translation_matrix((-0.15, -0.05, 0.2))),
        transform_shape(engine2, translation_matrix((0.15, -0.05, 0.2)))
    )

    return concat_shapes(body, engines)

@register()
def bridge_segment(length: float) -> Shape:
    # Create main bridge structure
    bridge = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.1,
                                      'p0': (0, 0, 0),
                                      'p1': (length, 0, 0)},
                          color=(0.3, 0.3, 0.4))

    # Add neon lighting
    lights = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.02,
                                      'p0': (0, 0.12, 0),
                                      'p1': (length, 0.12, 0)},
                          color=(0, 1, 1))  # Cyan neon

    return concat_shapes(bridge, lights)

@register()
def city_district(size: float) -> Shape:
    def building_loop(i):
        height = np.random.uniform(2, 8)
        width = np.random.uniform(0.5, 1.5)
        depth = np.random.uniform(0.5, 1.5)

        building = library_call('skyscraper', width=width, height=height, depth=depth)
        x_pos = np.random.uniform(-size/2, size/2)
        z_pos = np.random.uniform(-size/2, size/2)

        return transform_shape(building, translation_matrix((x_pos, 0, z_pos)))

    return loop(int(size * 3), building_loop)

@register()
def aerial_traffic(bounds: float) -> Shape:
    def vehicle_loop(i):
        vehicle = library_call('flying_vehicle')
        x = np.random.uniform(-bounds, bounds)
        y = np.random.uniform(3, 10)
        z = np.random.uniform(-bounds, bounds)

        # Random rotation for varied directions
        angle = np.random.uniform(0, 2 * math.pi)

        vehicle = transform_shape(vehicle, translation_matrix((x, y, z)))
        return transform_shape(vehicle, rotation_matrix(angle, (0, 1, 0), compute_shape_center(vehicle)))

    return loop(20, vehicle_loop)

@register()
def bridge_network(size: float) -> Shape:
    def bridge_loop(i):
        length = np.random.uniform(3, 8)
        x = np.random.uniform(-size/2, size/2)
        y = np.random.uniform(2, 6)
        z = np.random.uniform(-size/2, size/2)

        bridge = library_call('bridge_segment', length=length)
        bridge = transform_shape(bridge, translation_matrix((x, y, z)))

        # Random rotation around y-axis
        angle = np.random.uniform(0, 2 * math.pi)
        return transform_shape(bridge, rotation_matrix(angle, (0, 1, 0), compute_shape_center(bridge)))

    return loop(int(size * 2), bridge_loop)

@register()
def futuristic_cityscape() -> Shape:
    # Create main city districts
    city = library_call('city_district', size=20)

    # Add flying vehicles
    vehicles = library_call('aerial_traffic', bounds=10)

    # Add connecting bridges
    bridges = library_call('bridge_network', size=20)

    return concat_shapes(city, vehicles, bridges)
"""

This implementation creates a futuristic cityscape with:

1. Skyscrapers with randomly placed illuminated windows
2. Flying vehicles with glowing engines
3. A network of bridges with neon lighting
4. Multiple city districts with buildings of varying heights
5. Aerial traffic with vehicles moving in different directions

The cityscape features:
- Randomized building placements and sizes
- Varied heights and dimensions for visual interest
- Glowing elements (windows, vehicle engines, bridge lights) to create a nighttime atmosphere
- Multiple layers (ground level buildings, mid-level bridges, high-level aerial traffic)
- Color scheme using dark bases with bright accent lights

You can create the entire cityscape by calling:
```python
cityscape = library_call('futuristic_cityscape')
```
"""