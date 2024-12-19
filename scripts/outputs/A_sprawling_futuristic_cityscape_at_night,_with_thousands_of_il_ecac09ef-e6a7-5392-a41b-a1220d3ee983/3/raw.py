"""
I'll help create a futuristic cityscape with modular components. Here's my implementation:

"""
from helper import *

"""
A sprawling futuristic cityscape at night, with thousands of illuminated buildings,
flying vehicles, and interconnected bridges, all dynamically lit by neon lights
and holographic advertisements.
"""

@register()
def skyscraper(width: float, height: float, depth: float) -> Shape:
    # Create main building structure
    building = primitive_call('cube',
                            shape_kwargs={'scale': (width, height, depth)},
                            color=(0.2, 0.2, 0.3))  # Dark blue-grey base

    # Add glowing windows
    window_size = min(width, depth) * 0.1
    def window_fn(i) -> Shape:
        window = primitive_call('cube',
                              shape_kwargs={'scale': (window_size, window_size, window_size)},
                              color=(1, 1, 0.8))  # Warm light
        x_offset = np.random.uniform(-width/2 + window_size, width/2 - window_size)
        y_offset = np.random.uniform(-height/2 + window_size, height/2 - window_size)
        z_offset = depth/2  # Windows on the front face
        return transform_shape(window, translation_matrix((x_offset, y_offset, z_offset)))

    windows = loop(int(height * width * 2), window_fn)
    return concat_shapes(building, windows)

@register()
def flying_vehicle() -> Shape:
    # Create sleek vehicle body
    body = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.1,
                                     'p0': (-0.2, 0, 0),
                                     'p1': (0.2, 0, 0)},
                         color=(0.7, 0, 1))  # Purple glow

    # Add engine glow
    engine = primitive_call('sphere',
                          shape_kwargs={'radius': 0.05},
                          color=(1, 0.5, 0))  # Orange thruster
    engine = transform_shape(engine, translation_matrix((-0.2, 0, 0)))

    return concat_shapes(body, engine)

@register()
def bridge_segment(length: float) -> Shape:
    # Create main bridge structure
    bridge = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.05,
                                      'p0': (0, 0, 0),
                                      'p1': (length, 0, 0)},
                          color=(0, 0.8, 1))  # Cyan glow
    return bridge

@register()
def district(size: float) -> Shape:
    def building_fn(i) -> Shape:
        height = np.random.uniform(size*0.5, size*2)
        width = np.random.uniform(size*0.1, size*0.3)
        depth = np.random.uniform(size*0.1, size*0.3)

        building = library_call('skyscraper', width=width, height=height, depth=depth)
        x = np.random.uniform(-size/2, size/2)
        z = np.random.uniform(-size/2, size/2)

        return transform_shape(building, translation_matrix((x, height/2, z)))

    buildings = loop(15, building_fn)

    # Add connecting bridges
    def bridge_fn(i) -> Shape:
        length = np.random.uniform(size*0.2, size*0.4)
        bridge = library_call('bridge_segment', length=length)

        x = np.random.uniform(-size/2, size/2)
        y = np.random.uniform(size*0.3, size)
        z = np.random.uniform(-size/2, size/2)

        angle = np.random.uniform(0, 2*math.pi)
        return transform_shape(bridge,
                             translation_matrix((x, y, z)) @
                             rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))

    bridges = loop(10, bridge_fn)

    # Add flying vehicles
    def vehicle_fn(i) -> Shape:
        vehicle = library_call('flying_vehicle')
        x = np.random.uniform(-size/2, size/2)
        y = np.random.uniform(size*0.2, size*1.5)
        z = np.random.uniform(-size/2, size/2)

        angle = np.random.uniform(0, 2*math.pi)
        return transform_shape(vehicle,
                             translation_matrix((x, y, z)) @
                             rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))

    vehicles = loop(8, vehicle_fn)

    return concat_shapes(buildings, bridges, vehicles)

@register()
def cityscape() -> Shape:
    # Create multiple districts
    def district_fn(i) -> Shape:
        district_size = 5.0
        x_offset = ((i % 3) - 1) * district_size * 1.2
        z_offset = ((i // 3) - 1) * district_size * 1.2

        district_shape = library_call('district', size=district_size)
        return transform_shape(district_shape, translation_matrix((x_offset, 0, z_offset)))

    return loop(9, district_fn)  # 3x3 grid of districts
"""

This implementation creates a futuristic cityscape with:

1. `skyscraper`: Buildings with randomly placed glowing windows
2. `flying_vehicle`: Sleek vehicles with glowing engines
3. `bridge_segment`: Illuminated bridges connecting buildings
4. `district`: A collection of buildings, bridges, and vehicles
5. `cityscape`: Multiple districts arranged in a 3x3 grid

The design features:
- Randomized building heights and positions
- Glowing elements using bright colors against dark bases
- Multiple layers of detail (windows, vehicles, bridges)
- Interconnected elements creating depth and complexity
- Dynamic positioning and rotation of elements
- Scale hierarchy from individual buildings to entire districts

The cityscape uses a cool color palette (blues, purples, cyans) with warm accents (yellow windows, orange engine glows) to create a nighttime cyberpunk atmosphere.
"""