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
    def window_fn(i: int) -> Shape:
        window = primitive_call('cube',
                              shape_kwargs={'scale': (window_size, window_size, window_size)},
                              color=(1, 1, 0.8))
        x_offset = np.random.uniform(-width/2 + window_size, width/2 - window_size)
        z_offset = np.random.uniform(-depth/2 + window_size, depth/2 - window_size)
        y_offset = np.random.uniform(-height/2 + window_size, height/2 - window_size)
        return transform_shape(window, translation_matrix((x_offset, y_offset, z_offset)))

    windows = loop(int(height * 5), window_fn)
    return concat_shapes(building, windows)

@register()
def flying_vehicle() -> Shape:
    # Futuristic vehicle made of connected spheres
    body = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(0.7, 0, 1))
    wing1 = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.02, 'p0': (0.1, 0, 0), 'p1': (0.2, 0, 0.1)},
                          color=(1, 0, 0.5))
    wing2 = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.02, 'p0': (-0.1, 0, 0), 'p1': (-0.2, 0, 0.1)},
                          color=(1, 0, 0.5))
    return concat_shapes(body, wing1, wing2)

@register()
def bridge_segment(length: float) -> Shape:
    main_beam = primitive_call('cylinder',
                             shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (length, 0, 0)},
                             color=(0, 0.8, 1))
    glow = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.02, 'p0': (0, -0.05, 0), 'p1': (length, -0.05, 0)},
                         color=(0, 1, 1))
    return concat_shapes(main_beam, glow)

@register()
def district(size: float, num_buildings: int) -> Shape:
    def building_fn(i: int) -> Shape:
        height = np.random.uniform(1, 4)
        width = np.random.uniform(0.2, 0.5)
        depth = np.random.uniform(0.2, 0.5)
        building = library_call('skyscraper', width=width, height=height, depth=depth)

        x = np.random.uniform(-size/2, size/2)
        z = np.random.uniform(-size/2, size/2)
        return transform_shape(building, translation_matrix((x, 0, z)))

    return loop(num_buildings, building_fn)

@register()
def aerial_traffic(bounds: float, num_vehicles: int) -> Shape:
    def vehicle_fn(i: int) -> Shape:
        vehicle = library_call('flying_vehicle')
        x = np.random.uniform(-bounds, bounds)
        y = np.random.uniform(1, 3)
        z = np.random.uniform(-bounds, bounds)
        angle = np.random.uniform(0, 2*math.pi)

        vehicle = transform_shape(vehicle, translation_matrix((x, y, z)))
        center = compute_shape_center(vehicle)
        return transform_shape(vehicle, rotation_matrix(angle, (0, 1, 0), center))

    return loop(num_vehicles, vehicle_fn)

@register()
def bridge_network(size: float, num_bridges: int) -> Shape:
    def bridge_fn(i: int) -> Shape:
        length = np.random.uniform(1, 3)
        bridge = library_call('bridge_segment', length=length)

        x = np.random.uniform(-size/2, size/2)
        y = np.random.uniform(0.5, 2)
        z = np.random.uniform(-size/2, size/2)
        angle = np.random.uniform(0, 2*math.pi)

        bridge = transform_shape(bridge, translation_matrix((x, y, z)))
        center = compute_shape_center(bridge)
        return transform_shape(bridge, rotation_matrix(angle, (0, 1, 0), center))

    return loop(num_bridges, bridge_fn)

@register()
def futuristic_city() -> Shape:
    # Create main districts of buildings
    central_district = library_call('district', size=5, num_buildings=50)
    outer_district = library_call('district', size=8, num_buildings=30)

    # Add flying vehicles
    traffic = library_call('aerial_traffic', bounds=4, num_vehicles=20)

    # Add connecting bridges
    bridges = library_call('bridge_network', size=5, num_bridges=15)

    return concat_shapes(central_district, outer_district, traffic, bridges)
"""

This implementation creates a futuristic cityscape with:

1. Skyscrapers with randomly placed glowing windows
2. Flying vehicles with neon-colored wings
3. A network of illuminated bridges connecting different parts of the city
4. Multiple districts with buildings of varying heights
5. Aerial traffic with vehicles at different heights and orientations

The city is built modularly, with each component (skyscraper, vehicle, bridge) defined separately and then combined into districts and finally the complete cityscape. The buildings feature glowing windows, the vehicles have distinctive neon-colored wings, and the bridges have glowing elements, all contributing to the futuristic night-time atmosphere.
"""