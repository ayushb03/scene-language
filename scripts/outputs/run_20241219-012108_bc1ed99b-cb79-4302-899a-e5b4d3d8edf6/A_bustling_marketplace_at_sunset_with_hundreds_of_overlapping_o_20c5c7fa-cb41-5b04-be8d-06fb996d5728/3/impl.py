

from pathlib import Path
import numpy as np

from helper import *
import mitsuba as mi
import traceback
import ipdb

import random
import math
import sys
import os

import mi_helper  # such that primitive call will be implemented


def main():
    from example_postprocess import parse_program
    from engine.utils.graph_utils import strongly_connected_components, get_root
    # from tu.train_setup import set_seed
    from engine.utils.train_utils import set_seed
    from dsl_utils import library, animation_func
    from minecraft_helper import execute, execute_animation

    set_seed(0)

    save_dir = Path(__file__).parent / 'renderings'
    save_dir.mkdir(exist_ok=True)

    if animation_func:
        frames = list(animation_func())
        name = animation_func.__name__
        execute_animation(frames, save_dir=(save_dir / name).as_posix(), description=name)
    else:
        exp_program_path = Path(__file__).parent / 'program.py'
        _, library_equiv = parse_program(exp_program_path.as_posix())
        scc = strongly_connected_components(library_equiv)
        print(f'{scc=}')

        try:
            root = get_root(library_equiv)
            print(f'{root=}')
        except Exception as e:
            # sometimes a function is implemented but never used, so there is no shared ancestor
            root = None
            print('[ERROR] cannot find root')
            for name, node in library_equiv.items():
                if len(node.parents) == 0 and len(node.children) > 0:
                    root = name
            if root is None:  # not sure, just pick anything?
                root = next(reversed(library.keys()))
            print(e)

        node = library_equiv[root]
        execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)

        save_dir = Path(__file__).parent / 'extra_renderings'
        for node in library_equiv.values():
            try:
                execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)
            except:
                import traceback; traceback.print_exc()
                pass


"""
I'll help create a marketplace scene with various stalls, decorations, and structures. I'll break it down into modular components:

"""
from helper import *

@register()
def marketplace() -> Shape:
    return concat_shapes(
        library_call("ground_platform"),
        library_call("stall_row", num_stalls=5),
        transform_shape(
            library_call("stall_row", num_stalls=4),
            translation_matrix([2, 0, 8])
        ),
        library_call("lantern_array"),
        library_call("decorative_elements")
    )

@register()
def ground_platform() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(20, 1, 15),
        fill=True
    )

@register()
def basic_stall() -> Shape:
    # Create a stall with roof and counter
    return concat_shapes(
        # Counter
        primitive_call(
            "set_cuboid",
            block_type="minecraft:oak_planks",
            scale=(3, 1, 2),
            fill=True
        ),
        # Back wall
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:oak_planks",
                scale=(3, 3, 1),
                fill=True
            ),
            translation_matrix([0, 1, 0])
        ),
        # Roof
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:dark_oak_stairs",
                block_kwargs={"facing": "north", "half": "bottom"},
                scale=(3, 1, 2),
                fill=True
            ),
            translation_matrix([0, 4, 0])
        )
    )

@register()
def stall_row(num_stalls: int = 5) -> Shape:
    def create_stall(i: int) -> Shape:
        return transform_shape(
            library_call("decorated_stall"),
            translation_matrix([i * 4, 0, 0])
        )
    return loop(num_stalls, create_stall)

@register()
def decorated_stall() -> Shape:
    return concat_shapes(
        library_call("basic_stall"),
        # Add items on counter
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:barrel",
                block_kwargs={"facing": "up"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([1, 1, 1])
        ),
        # Add banner
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:red_banner",
                block_kwargs={"rotation": "8"},
                scale=(1, 2, 1),
                fill=True
            ),
            translation_matrix([1, 2, 0])
        )
    )

@register()
def lantern_array() -> Shape:
    def place_lantern(i: int) -> Shape:
        x = (i % 4) * 5
        z = (i // 4) * 5
        return transform_shape(
            library_call("hanging_lantern"),
            translation_matrix([x + 2, 0, z + 2])
        )
    return loop(12, place_lantern)

@register()
def hanging_lantern() -> Shape:
    return concat_shapes(
        # Chain
        primitive_call(
            "set_cuboid",
            block_type="minecraft:chain",
            block_kwargs={"axis": "y"},
            scale=(1, 2, 1),
            fill=True
        ),
        # Lantern
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:lantern",
                block_kwargs={"hanging": "true"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 0, 0])
        )
    )

@register()
def decorative_elements() -> Shape:
    def place_decoration(i: int) -> Shape:
        x = (i % 5) * 4
        z = (i // 5) * 4
        if i % 3 == 0:
            block = "minecraft:potted_red_tulip"
        elif i % 3 == 1:
            block = "minecraft:potted_bamboo"
        else:
            block = "minecraft:potted_azure_bluet"

        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type=block,
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x + 1, 1, z + 4])
        )
    return loop(15, place_decoration)
"""

This code creates a marketplace scene with:
1. A stone platform as the base
2. Two rows of market stalls with varying decorations
3. Hanging lanterns spread throughout the marketplace
4. Decorative elements like potted plants scattered around

Each stall has:
- A wooden counter and back wall
- A sloped roof made of stairs
- Decorative elements like barrels and banners
- Unique spacing and positioning

The scene includes:
- Multiple hanging lanterns for lighting
- Decorative elements like potted plants
- A organized layout with proper spacing between elements

The marketplace is designed to be visually interesting while maintaining reasonable size constraints and using valid Minecraft blocks. The modular approach allows for easy modification and expansion of the scene.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
