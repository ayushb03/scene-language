

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
I'll help create a marketplace scene with various stalls, decorations, and structures. I'll break this down into modular components:

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
    # Create a stall with a roof and counter
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
def lantern_array() -> Shape:
    def place_lantern(i):
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:lantern",
                block_kwargs={"hanging": "true"},
                scale=(1, 1, 1),
                fill=True
            ),
            translation_matrix([i * 4, 5, i % 2 * 4 + 2])
        )
    return loop(5, place_lantern)

@register()
def stall_row(num_stalls: int = 5) -> Shape:
    def place_stall(i):
        return transform_shape(
            library_call("decorated_stall"),
            translation_matrix([i * 4, 0, 0])
        )
    return loop(num_stalls, place_stall)

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
        # Add banner decoration
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
def decorative_elements() -> Shape:
    def place_decoration(i):
        return concat_shapes(
            # Place some flower pots
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:potted_red_tulip",
                    scale=(1, 1, 1),
                    fill=True
                ),
                translation_matrix([i * 5, 1, 14])
            ),
            # Add some fence posts
            transform_shape(
                primitive_call(
                    "set_cuboid",
                    block_type="minecraft:oak_fence",
                    scale=(1, 2, 1),
                    fill=True
                ),
                translation_matrix([i * 5, 0, 13])
            )
        )
    return loop(4, place_decoration)
"""

This implementation creates:
1. A ground platform made of smooth stone
2. Two rows of market stalls with varying decorations
3. Hanging lanterns arranged in a pattern above the stalls
4. Decorative elements including flower pots and fence posts
5. Each stall has:
   - A counter made of oak planks
   - A back wall
   - A sloped roof using stairs
   - Decorative elements like barrels and banners
   - Spacing between stalls for movement

The scene creates depth through multiple rows of stalls and varying heights of decorative elements. The lanterns provide ambient lighting effects, while the banners and flower pots add color and life to the scene.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
