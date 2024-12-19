

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
        library_call("ground_floor"),
        library_call("stall_row", num_stalls=5),
        transform_shape(
            library_call("stall_row", num_stalls=4),
            translation_matrix([2, 0, 8])
        ),
        library_call("lantern_array"),
        library_call("crowd_simulation"),
        library_call("kite_display")
    )

@register()
def ground_floor() -> Shape:
    # Create stone brick foundation
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(20, 1, 15),
        fill=True
    )

@register()
def single_stall() -> Shape:
    # Create a food stall with wooden frame and colorful awning
    return concat_shapes(
        # Wooden frame
        primitive_call(
            "set_cuboid",
            block_type="minecraft:oak_fence",
            scale=(3, 3, 3),
            fill=False
        ),
        # Counter
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:barrel",
                scale=(3, 1, 1),
                fill=True
            ),
            translation_matrix([0, 1, 0])
        ),
        # Awning
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:red_wool",
                scale=(3, 1, 3),
                fill=True
            ),
            translation_matrix([0, 3, 0])
        )
    )

@register()
def stall_row(num_stalls: int = 5) -> Shape:
    def create_stall(i: int) -> Shape:
        return transform_shape(
            library_call("single_stall"),
            translation_matrix([i * 4, 0, 0])
        )
    return loop(num_stalls, create_stall)

@register()
def hanging_lantern() -> Shape:
    return concat_shapes(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:lantern",
            block_kwargs={"hanging": "true"},
            scale=(1, 1, 1),
            fill=True
        ),
        primitive_call(
            "set_cuboid",
            block_type="minecraft:chain",
            scale=(1, 2, 1),
            fill=True
        )
    )

@register()
def lantern_array() -> Shape:
    def place_lantern(i: int) -> Shape:
        x = (i % 4) * 5
        z = (i // 4) * 5
        return transform_shape(
            library_call("hanging_lantern"),
            translation_matrix([x + 2, 5, z + 2])
        )
    return loop(12, place_lantern)

@register()
def person() -> Shape:
    # Simple person representation using concrete blocks
    return concat_shapes(
        # Body
        primitive_call(
            "set_cuboid",
            block_type="minecraft:gray_concrete",
            scale=(1, 2, 1),
            fill=True
        ),
        # Head
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:pink_concrete",
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([0, 2, 0])
        )
    )

@register()
def crowd_simulation() -> Shape:
    def place_person(i: int) -> Shape:
        x = (i % 7) * 2 + 3
        z = (i // 7) * 2 + 3
        return transform_shape(
            library_call("person"),
            translation_matrix([x, 1, z])
        )
    return loop(15, place_person)

@register()
def kite() -> Shape:
    return concat_shapes(
        # Kite diamond
        primitive_call(
            "set_cuboid",
            block_type="minecraft:light_blue_stained_glass",
            scale=(2, 2, 1),
            fill=True
        ),
        # Kite tail
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:white_wool",
                scale=(1, 3, 1),
                fill=True
            ),
            translation_matrix([0, -3, 0])
        )
    )

@register()
def kite_display() -> Shape:
    def place_kite(i: int) -> Shape:
        x = (i % 3) * 4 + 5
        z = (i // 3) * 4 + 2
        height = 6 + (i % 2) * 2  # Varying heights
        return transform_shape(
            library_call("kite"),
            translation_matrix([x, height, z])
        )
    return loop(4, place_kite)
"""

This implementation creates a lively marketplace scene with:

1. A stone ground floor as the foundation
2. Two rows of market stalls with wooden frames, counters, and colorful awnings
3. An array of hanging lanterns providing ambient lighting
4. A crowd of people represented by simple block figures
5. Floating kites at different heights to add movement and playfulness

The scene is modular and uses nested components to create complexity. The marketplace has:
- Multiple stalls arranged in rows
- Lanterns hanging at regular intervals
- A crowd of people scattered around
- Kites floating at different heights

Each component is built using primitive shapes and transformed to their correct positions. The scene maintains reasonable dimensions while creating the impression of a busy marketplace.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
