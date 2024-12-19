

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
I'll help create a marketplace scene with various elements. I'll break it down into modular components:

"""
from helper import *

@register()
def marketplace() -> Shape:
    return concat_shapes(
        library_call("ground_plaza"),
        library_call("stall_row", num_stalls=5),
        transform_shape(
            library_call("stall_row", num_stalls=4),
            translation_matrix([2, 0, 8])
        ),
        library_call("lantern_array"),
        library_call("decorative_elements")
    )

@register()
def ground_plaza() -> Shape:
    # Create stone brick foundation
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(15, 1, 12),
        fill=True
    )

@register()
def stall(variant: int = 0) -> Shape:
    # Basic stall structure
    base = primitive_call(
        "set_cuboid",
        block_type="minecraft:oak_planks",
        scale=(3, 3, 2),
        fill=False
    )

    # Roof made of stairs
    roof = primitive_call(
        "set_cuboid",
        block_type="minecraft:dark_oak_stairs",
        block_kwargs={"facing": "north", "half": "bottom"},
        scale=(3, 1, 2),
        fill=True
    )

    # Counter
    counter = primitive_call(
        "set_cuboid",
        block_type="minecraft:stripped_oak_log",
        block_kwargs={"axis": "x"},
        scale=(3, 1, 1),
        fill=True
    )

    return concat_shapes(
        base,
        transform_shape(roof, translation_matrix([0, 3, 0])),
        transform_shape(counter, translation_matrix([0, 1, 0]))
    )

@register()
def stall_row(num_stalls: int = 5) -> Shape:
    def create_stall(i: int) -> Shape:
        return transform_shape(
            library_call("stall", variant=i % 3),
            translation_matrix([i * 3, 0, 0])
        )
    return loop(num_stalls, create_stall)

@register()
def lantern() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:lantern",
        block_kwargs={"hanging": "true"},
        scale=(1, 1, 1),
        fill=True
    )

@register()
def lantern_array() -> Shape:
    def place_lantern(i: int) -> Shape:
        x = (i % 5) * 3
        z = (i // 5) * 3
        return transform_shape(
            library_call("lantern"),
            translation_matrix([x + 1, 4, z + 1])
        )
    return loop(15, place_lantern)

@register()
def decorative_elements() -> Shape:
    return concat_shapes(
        library_call("market_stands"),
        library_call("ambient_items")
    )

@register()
def market_stands() -> Shape:
    def create_stand(i: int) -> Shape:
        items = primitive_call(
            "set_cuboid",
            block_type="minecraft:barrel",
            block_kwargs={"facing": "up"},
            scale=(1, 1, 1),
            fill=True
        )
        return transform_shape(
            items,
            translation_matrix([i * 2 + 1, 1, 1])
        )
    return loop(6, create_stand)

@register()
def ambient_items() -> Shape:
    def place_item(i: int) -> Shape:
        x = (i % 7) * 2
        z = (i // 7) * 2
        block_types = ["minecraft:potted_red_tulip", "minecraft:cake", "minecraft:composter"]
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type=block_types[i % 3],
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([x + 1, 1, z + 3])
        )
    return loop(10, place_item)
"""

This implementation creates:
1. A ground plaza made of smooth stone
2. Two rows of market stalls with:
   - Wooden structure
   - Dark oak stairs for roofing
   - Counter made of stripped logs
3. Hanging lanterns in a grid pattern for lighting
4. Decorative elements including:
   - Barrels representing goods
   - Various items like flowers, cakes, and composters for ambiance

The marketplace is designed to be compact but detailed, with multiple layers of objects creating depth and interest. The stalls are modular and can be easily modified or expanded. The lanterns provide ambient lighting that would look especially nice at sunset, and the decorative elements add life and variety to the scene.

Note that while we can't literally create moving crowds or flying kites in Minecraft's static blocks, the arrangement of various items and structures creates a sense of a busy marketplace through careful placement and variety of elements.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
