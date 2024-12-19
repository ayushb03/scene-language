"""
I'll help create a chessboard with chess pieces. I'll break it down into modular components:

"""
from helper import *

"""
a chessboard with a full set of chess pieces
"""

@register()
def pawn() -> Shape:
    """Creates a basic pawn using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 'p0': (0, 0.1, 0), 'p1': (0, 0.4, 0)})
    head = primitive_call('sphere', shape_kwargs={'radius': 0.12})
    head = transform_shape(head, translation_matrix((0, 0.5, 0)))
    return concat_shapes(base, body, head)

@register()
def rook() -> Shape:
    """Creates a rook using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.12, 'p0': (0, 0.1, 0), 'p1': (0, 0.5, 0)})
    crown = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.15, 0.3)})
    crown = transform_shape(crown, translation_matrix((0, 0.6, 0)))
    return concat_shapes(base, body, crown)

@register()
def bishop() -> Shape:
    """Creates a bishop using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.11, 'p0': (0, 0.1, 0), 'p1': (0, 0.5, 0)})
    top = primitive_call('sphere', shape_kwargs={'radius': 0.08})
    top = transform_shape(top, translation_matrix((0, 0.65, 0)))
    return concat_shapes(base, body, top)

@register()
def knight() -> Shape:
    """Creates a knight using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.11, 'p0': (0, 0.1, 0), 'p1': (0, 0.4, 0)})
    head = primitive_call('cube', shape_kwargs={'scale': (0.15, 0.3, 0.25)})
    head = transform_shape(head, translation_matrix((0, 0.55, 0.05)))
    return concat_shapes(base, body, head)

@register()
def queen() -> Shape:
    """Creates a queen using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.17, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.1, 0), 'p1': (0, 0.6, 0)})
    crown = primitive_call('sphere', shape_kwargs={'radius': 0.15})
    crown = transform_shape(crown, translation_matrix((0, 0.75, 0)))
    return concat_shapes(base, body, crown)

@register()
def king() -> Shape:
    """Creates a king using primitive shapes"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.17, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.1, 0), 'p1': (0, 0.7, 0)})
    cross_v = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, 0.1)})
    cross_h = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.1, 0.1)})
    cross_v = transform_shape(cross_v, translation_matrix((0, 0.9, 0)))
    cross_h = transform_shape(cross_h, translation_matrix((0, 0.85, 0)))
    return concat_shapes(base, body, cross_v, cross_h)

@register()
def chess_square(is_white: bool) -> Shape:
    """Creates a single chess square"""
    color = (0.9, 0.9, 0.9) if is_white else (0.2, 0.2, 0.2)
    return primitive_call('cube', shape_kwargs={'scale': (1.0, 0.1, 1.0)}, color=color)

@register()
def chessboard() -> Shape:
    """Creates the 8x8 chessboard"""
    def loop_fn(i: int) -> Shape:
        row = i // 8
        col = i % 8
        square = library_call('chess_square', is_white=((row + col) % 2 == 0))
        return transform_shape(square, translation_matrix((col - 3.5, 0, row - 3.5)))
    return loop(64, loop_fn)

@register()
def piece_set(is_white: bool) -> Shape:
    """Creates a set of chess pieces for one side"""
    color = (0.9, 0.9, 0.9) if is_white else (0.3, 0.3, 0.3)
    pieces = []
    back_row = [
        (library_call('rook'), -3.5),
        (library_call('knight'), -2.5),
        (library_call('bishop'), -1.5),
        (library_call('queen'), -0.5),
        (library_call('king'), 0.5),
        (library_call('bishop'), 1.5),
        (library_call('knight'), 2.5),
        (library_call('rook'), 3.5)
    ]

    # Place back row
    z_pos = 3.5 if is_white else -3.5
    for piece, x_pos in back_row:
        piece = transform_shape(piece, translation_matrix((x_pos, 0.1, z_pos)))
        pieces.append(transform_shape(piece, scale_matrix(0.8, (x_pos, 0, z_pos))))

    # Place pawns
    z_pos = 2.5 if is_white else -2.5
    for x_pos in range(-4, 4):
        pawn_piece = library_call('pawn')
        pawn_piece = transform_shape(pawn_piece, translation_matrix((x_pos + 0.5, 0.1, z_pos)))
        pieces.append(transform_shape(pawn_piece, scale_matrix(0.7, (x_pos + 0.5, 0, z_pos))))

    return concat_shapes(*[transform_shape(p, scale_matrix(1.0, (0, 0, 0))) for p in pieces])

@register()
def complete_chess_set() -> Shape:
    """Creates the complete chessboard with all pieces"""
    board = library_call('chessboard')
    white_pieces = library_call('piece_set', is_white=True)
    black_pieces = library_call('piece_set', is_white=False)

    return concat_shapes(
        board,
        white_pieces,
        black_pieces
    )
"""

This program creates a complete chess set with:
1. A modular design where each piece type is a separate function
2. Pieces are constructed using basic primitives (cylinders, spheres, and cubes)
3. The chessboard is created using alternating black and white squares
4. Pieces are properly scaled and positioned on the board
5. The complete set includes:
   - 8x8 checkered board
   - 16 white pieces (8 pawns, 2 rooks, 2 knights, 2 bishops, 1 queen, 1 king)
   - 16 black pieces (same composition)

The pieces are simplified representations but recognizable:
- Pawns: cylindrical base with spherical top
- Rooks: cylindrical base with cubic crown
- Bishops: cylindrical base with small spherical top
- Knights: cylindrical base with rectangular head
- Queens: taller with larger spherical crown
- Kings: tallest with cross on top

The board is centered at the origin with squares of size 1x1, and pieces are scaled appropriately to fit on the squares.
"""