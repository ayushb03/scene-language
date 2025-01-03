"""
I'll create a futuristic cityscape with modular components. Here's my implementation:

"""
from helper import *

"""
A sprawling futuristic cityscape at night, with thousands of illuminated buildings,
flying vehicles, and interconnected bridges, all dynamically lit by neon lights.
"""

@register()
def skyscraper(width: float, height: float, depth: float) -> Shape:
    # Base building
    building = primitive_call('cube',
                            shape_kwargs={'scale': (width, height, depth)},
                            color=(0.2, 0.2, 0.3))

    # Add glowing windows
    window_size = min(width, depth) * 0.1
    def window_fn(i) -> Shape:
        window = primitive_call('cube',
                              shape_kwargs={'scale': (window_size, window_size, window_size)},
                              color=(1, 1, 0.8))
        x_pos = np.random.uniform(-width/2 + window_size, width/2 - window_size)
        z_pos = np.random.uniform(-depth/2 + window_size, depth/2 - window_size)
        y_pos = np.random.uniform(-height/2 + window_size, height/2 - window_size)
        return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

    windows = loop(int(height * 5), window_fn)
    return concat_shapes(building, windows)

@register()
def flying_vehicle() -> Shape:
    # Futuristic vehicle made of connected spheres
    body = primitive_call('sphere',
                         shape_kwargs={'radius': 0.3},
                         color=(0.7, 0.1, 0.1))

    wing1 = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.05,
                                      'p0': (-0.4, 0, 0),
                                      'p1': (0.4, 0, 0)},
                          color=(0.8, 0.2, 0.2))

    engine_glow = primitive_call('sphere',
                                shape_kwargs={'radius': 0.1},
                                color=(0, 1, 1))

    engine_glow = transform_shape(engine_glow, translation_matrix((0, 0, 0.4)))

    return concat_shapes(body, wing1, engine_glow)

@register()
def bridge_segment(length: float) -> Shape:
    main_beam = primitive_call('cylinder',
                             shape_kwargs={'radius': 0.1,
                                         'p0': (0, 0, 0),
                                         'p1': (length, 0, 0)},
                             color=(0.3, 0.3, 0.4))

    neon_strip = primitive_call('cylinder',
                              shape_kwargs={'radius': 0.02,
                                          'p0': (0, 0.15, 0),
                                          'p1': (length, 0.15, 0)},
                              color=(1, 0, 1))

    return concat_shapes(main_beam, neon_strip)

@register()
def district(size: float, num_buildings: int) -> Shape:
    def building_fn(i) -> Shape:
        height = np.random.uniform(size*0.5, size*2)
        width = np.random.uniform(size*0.2, size*0.4)
        depth = np.random.uniform(size*0.2, size*0.4)

        building = library_call('skyscraper', width=width, height=height, depth=depth)
        x_pos = np.random.uniform(-size/2, size/2)
        z_pos = np.random.uniform(-size/2, size/2)

        return transform_shape(building, translation_matrix((x_pos, height/2, z_pos)))

    return loop(num_buildings, building_fn)

@register()
def vehicles_layer(size: float, num_vehicles: int) -> Shape:
    def vehicle_fn(i) -> Shape:
        vehicle = library_call('flying_vehicle')
        x_pos = np.random.uniform(-size, size)
        y_pos = np.random.uniform(size*0.3, size)
        z_pos = np.random.uniform(-size, size)

        vehicle = transform_shape(vehicle, translation_matrix((x_pos, y_pos, z_pos)))
        return transform_shape(vehicle,
                             rotation_matrix(np.random.uniform(0, 2*np.pi),
                                          direction=(0, 1, 0),
                                          point=compute_shape_center(vehicle)))

    return loop(num_vehicles, vehicle_fn)

@register()
def bridge_network(size: float, num_bridges: int) -> Shape:
    def bridge_fn(i) -> Shape:
        length = np.random.uniform(size*0.3, size*0.8)
        bridge = library_call('bridge_segment', length=length)

        x_pos = np.random.uniform(-size/2, size/2)
        y_pos = np.random.uniform(size*0.2, size*0.6)
        z_pos = np.random.uniform(-size/2, size/2)

        bridge = transform_shape(bridge, translation_matrix((x_pos, y_pos, z_pos)))
        return transform_shape(bridge,
                             rotation_matrix(np.random.uniform(0, 2*np.pi),
                                          direction=(0, 1, 0),
                                          point=compute_shape_center(bridge)))

    return loop(num_bridges, bridge_fn)

@register()
def futuristic_city() -> Shape:
    # Create main city components
    city_size = 20.0
    buildings = library_call('district', size=city_size, num_buildings=50)
    vehicles = library_call('vehicles_layer', size=city_size, num_vehicles=30)
    bridges = library_call('bridge_network', size=city_size, num_bridges=20)

    return concat_shapes(buildings, vehicles, bridges)
"""

This implementation creates a futuristic cityscape with:

1. Skyscrapers with randomly placed glowing windows
2. Flying vehicles with glowing engine effects
3. A network of bridges with neon lighting
4. Multiple districts of buildings at varying heights
5. Flying vehicles at different altitudes and orientations
6. Interconnected bridge network at various levels

The city is built modularly, with each component (skyscraper, vehicle, bridge) defined separately and then combined into larger structures (district, vehicles_layer, bridge_network). The final `futuristic_city` function combines everything into a complete cityscape.

Key features:
- Random variations in building sizes and positions
- Glowing effects using bright colors for windows and neon
- Flying vehicles with engine effects
- Multi-level bridge network with neon strips
- Varied heights and densities creating a dynamic skyline
- Randomized rotations and positions for vehicles and bridges

The scale and numbers can be adjusted in the `futuristic_city` function to create denser or sparser cityscapes as needed.
"""