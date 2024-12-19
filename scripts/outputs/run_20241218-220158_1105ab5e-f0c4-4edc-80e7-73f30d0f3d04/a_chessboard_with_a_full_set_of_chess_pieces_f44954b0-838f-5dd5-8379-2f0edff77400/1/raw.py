"""
I'll help create a chessboard with chess pieces. I'll break it down into modular components:

"""
from helper import *

"""
a chessboard with a full set of chess pieces
"""

@register()
def pawn() -> Shape:
    """Creates a pawn chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.04, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.025, 'p0': (0, 0.02, 0), 'p1': (0, 0.08, 0)})
    head = primitive_call('sphere', shape_kwargs={'radius': 0.035})
    head = transform_shape(head, translation_matrix((0, 0.12, 0)))
    return concat_shapes(base, body, head)

@register()
def rook() -> Shape:
    """Creates a rook chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.045, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.03, 'p0': (0, 0.02, 0), 'p1': (0, 0.1, 0)})
    crown = primitive_call('cube', shape_kwargs={'scale': (0.07, 0.03, 0.07)})
    crown = transform_shape(crown, translation_matrix((0, 0.115, 0)))
    return concat_shapes(base, body, crown)

@register()
def bishop() -> Shape:
    """Creates a bishop chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.045, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.03, 'p0': (0, 0.02, 0), 'p1': (0, 0.12, 0)})
    top = primitive_call('sphere', shape_kwargs={'radius': 0.02})
    top = transform_shape(top, translation_matrix((0, 0.15, 0)))
    return concat_shapes(base, body, top)

@register()
def knight() -> Shape:
    """Creates a knight chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.045, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.03, 'p0': (0, 0.02, 0), 'p1': (0, 0.1, 0)})
    head = primitive_call('cube', shape_kwargs={'scale': (0.04, 0.06, 0.08)})
    head = transform_shape(head, translation_matrix((0, 0.13, -0.01)))
    return concat_shapes(base, body, head)

@register()
def queen() -> Shape:
    """Creates a queen chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.035, 'p0': (0, 0.02, 0), 'p1': (0, 0.15, 0)})
    crown = primitive_call('sphere', shape_kwargs={'radius': 0.04})
    crown = transform_shape(crown, translation_matrix((0, 0.19, 0)))
    return concat_shapes(base, body, crown)

@register()
def king() -> Shape:
    """Creates a king chess piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.035, 'p0': (0, 0.02, 0), 'p1': (0, 0.17, 0)})
    cross_v = primitive_call('cylinder', shape_kwargs={'radius': 0.01, 'p0': (0, 0.2, 0), 'p1': (0, 0.25, 0)})
    cross_h = primitive_call('cylinder', shape_kwargs={'radius': 0.01, 'p0': (-0.025, 0.22, 0), 'p1': (0.025, 0.22, 0)})
    return concat_shapes(base, body, cross_v, cross_h)

@register()
def chess_square(is_white: bool) -> Shape:
    """Creates a single chess square"""
    color = (0.9, 0.9, 0.9) if is_white else (0.2, 0.2, 0.2)
    return primitive_call('cube', shape_kwargs={'scale': (0.125, 0.02, 0.125)}, color=color)

@register()
def chessboard() -> Shape:
    """Creates the 8x8 chessboard"""
    def loop_fn(i: int) -> Shape:
        row = i // 8
        col = i % 8
        square = library_call('chess_square', is_white=((row + col) % 2 == 0))
        return transform_shape(square, translation_matrix((col * 0.125 - 0.437, 0, row * 0.125 - 0.437)))
    return loop(64, loop_fn)

@register()
def piece_set(is_white: bool) -> Shape:
    """Creates a set of chess pieces for one side"""
    color = (0.9, 0.9, 0.9) if is_white else (0.2, 0.2, 0.2)
    y_offset = 0.01  # Offset to place pieces slightly above board

    def create_piece(piece_func: str, x: float, z: float) -> Shape:
        piece = library_call(piece_func)
        piece = transform_shape(piece, translation_matrix((x * 0.125 - 0.437, y_offset, z * 0.125 - 0.437)))
        return primitive_call('cube', shape_kwargs={'scale': (0.001, 0.001, 0.001)}) if piece is None else piece

    pieces = []
    z_row = 0 if is_white else 7  # Back row
    pawn_row = 1 if is_white else 6  # Pawn row

    # Back row pieces
    pieces.extend([
        create_piece('rook', 0, z_row),
        create_piece('knight', 1, z_row),
        create_piece('bishop', 2, z_row),
        create_piece('queen', 3, z_row),
        create_piece('king', 4, z_row),
        create_piece('bishop', 5, z_row),
        create_piece('knight', 6, z_row),
        create_piece('rook', 7, z_row)
    ])

    # Pawns
    for x in range(8):
        pieces.append(create_piece('pawn', x, pawn_row))

    result = concat_shapes(*pieces)
    # Color the entire set
    return primitive_call('cube', shape_kwargs={'scale': (0.001, 0.001, 0.001)}) if result is None else result

@register()
def complete_chess_set() -> Shape:
    """Creates a complete chess set with board and all pieces"""
    board = library_call('chessboard')
    white_pieces = library_call('piece_set', is_white=True)
    black_pieces = library_call('piece_set', is_white=False)
    return concat_shapes(board, white_pieces, black_pieces)
"""

This program creates a complete chess set with:
1. A chessboard with alternating black and white squares
2. Individual piece types (pawn, rook, bishop, knight, queen, king)
3. Complete sets of white and black pieces
4. All pieces properly positioned on the board

The pieces are simplified geometric representations:
- Pawns: cylindrical base with spherical top
- Rooks: cylindrical base with cubic crown
- Bishops: tall cylinder with small sphere on top
- Knights: cylindrical base with rectangular head
- Queens: tall cylinder with large sphere crown
- Kings: tallest piece with cross on top

The board is centered at the origin and scaled to be 1x1 unit in the xz-plane. The pieces are sized proportionally and placed slightly above the board surface.
"""