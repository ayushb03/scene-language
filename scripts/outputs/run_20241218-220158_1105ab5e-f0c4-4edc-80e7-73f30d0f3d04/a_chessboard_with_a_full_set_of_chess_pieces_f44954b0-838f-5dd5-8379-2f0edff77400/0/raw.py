"""
I'll help create a chessboard with chess pieces. I'll break it down into modular components:

"""
from helper import *

"""
a chessboard with a full set of chess pieces
"""

@register()
def pawn() -> Shape:
    """Creates a basic pawn piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 'p0': (0, 0.1, 0), 'p1': (0, 0.4, 0)})
    head = primitive_call('sphere', shape_kwargs={'radius': 0.12})
    head = transform_shape(head, translation_matrix((0, 0.5, 0)))
    return concat_shapes(base, body, head)

@register()
def rook() -> Shape:
    """Creates a rook piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.12, 'p0': (0, 0.1, 0), 'p1': (0, 0.5, 0)})
    crown = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.15, 0.3)})
    crown = transform_shape(crown, translation_matrix((0, 0.6, 0)))
    return concat_shapes(base, body, crown)

@register()
def bishop() -> Shape:
    """Creates a bishop piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.11, 'p0': (0, 0.1, 0), 'p1': (0, 0.5, 0)})
    top = primitive_call('sphere', shape_kwargs={'radius': 0.08})
    top = transform_shape(top, translation_matrix((0, 0.65, 0)))
    return concat_shapes(base, body, top)

@register()
def knight() -> Shape:
    """Creates a knight piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.11, 'p0': (0, 0.1, 0), 'p1': (0, 0.4, 0)})
    head = primitive_call('cube', shape_kwargs={'scale': (0.15, 0.3, 0.2)})
    head = transform_shape(head, translation_matrix((0, 0.55, 0)))
    return concat_shapes(base, body, head)

@register()
def queen() -> Shape:
    """Creates a queen piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.17, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.1, 0), 'p1': (0, 0.6, 0)})
    crown = primitive_call('sphere', shape_kwargs={'radius': 0.15})
    crown = transform_shape(crown, translation_matrix((0, 0.75, 0)))
    return concat_shapes(base, body, crown)

@register()
def king() -> Shape:
    """Creates a king piece"""
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
        ('rook', -3.5), ('knight', -2.5), ('bishop', -1.5), ('queen', -0.5),
        ('king', 0.5), ('bishop', 1.5), ('knight', 2.5), ('rook', 3.5)
    ]

    # Back row pieces
    for piece_name, x in back_row:
        piece = library_call(piece_name)
        z = -3.5 if is_white else 3.5
        piece = transform_shape(piece, translation_matrix((x, 0.05, z)))
        pieces.append(primitive_call('sphere', shape_kwargs={'radius': 0.01}, color=color))  # Color placeholder
        pieces.append(piece)

    # Pawns
    for x in range(-3, 5):
        pawn = library_call('pawn')
        z = -2.5 if is_white else 2.5
        pawn = transform_shape(pawn, translation_matrix((x - 0.5, 0.05, z)))
        pieces.append(primitive_call('sphere', shape_kwargs={'radius': 0.01}, color=color))  # Color placeholder
        pieces.append(pawn)

    return concat_shapes(*pieces)

@register()
def complete_chess_set() -> Shape:
    """Creates the complete chessboard with all pieces"""
    board = library_call('chessboard')
    white_pieces = library_call('piece_set', is_white=True)
    black_pieces = library_call('piece_set', is_white=False)
    return concat_shapes(board, white_pieces, black_pieces)
"""

This program creates a complete chess set with:
1. A modular design where each piece type is a separate function
2. Basic geometric shapes combined to create recognizable chess pieces
3. A chessboard with alternating black and white squares
4. Proper placement of all pieces in their starting positions
5. Two complete sets of pieces (black and white)

The pieces are simplified geometric representations:
- Pawns: cylinder base with sphere head
- Rooks: cylinder with cubic crown
- Bishops: cylinder with small sphere top
- Knights: cylinder with rectangular head
- Queens: taller cylinder with large sphere crown
- Kings: tallest piece with cross on top

The board is created using a loop to place 64 alternating squares, and pieces are positioned using translation matrices to their correct starting positions.
"""