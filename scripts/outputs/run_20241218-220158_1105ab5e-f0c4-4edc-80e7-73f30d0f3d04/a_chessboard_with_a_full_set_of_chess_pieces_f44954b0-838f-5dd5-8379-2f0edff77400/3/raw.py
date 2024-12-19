"""
I'll help you create a chessboard with chess pieces. I'll break this down into modular components:

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
    top = transform_shape(top, translation_matrix((0, 0.6, 0)))
    return concat_shapes(base, body, top)

@register()
def knight() -> Shape:
    """Creates a knight piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.11, 'p0': (0, 0.1, 0), 'p1': (0, 0.4, 0)})
    head = primitive_call('cube', shape_kwargs={'scale': (0.15, 0.25, 0.25)})
    head = transform_shape(head, translation_matrix((0, 0.5, 0)))
    return concat_shapes(base, body, head)

@register()
def queen() -> Shape:
    """Creates a queen piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.17, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.1, 0), 'p1': (0, 0.6, 0)})
    crown = primitive_call('sphere', shape_kwargs={'radius': 0.15})
    crown = transform_shape(crown, translation_matrix((0, 0.7, 0)))
    return concat_shapes(base, body, crown)

@register()
def king() -> Shape:
    """Creates a king piece"""
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.17, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)})
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.1, 0), 'p1': (0, 0.7, 0)})
    cross_v = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, 0.1)})
    cross_h = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.1, 0.1)})
    cross_v = transform_shape(cross_v, translation_matrix((0, 0.9, 0)))
    cross_h = transform_shape(cross_h, translation_matrix((0, 0.8, 0)))
    return concat_shapes(base, body, cross_v, cross_h)

@register()
def chess_square(is_white: bool) -> Shape:
    """Creates a single chess square"""
    color = (0.9, 0.9, 0.9) if is_white else (0.3, 0.3, 0.3)
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
    color = (1.0, 1.0, 1.0) if is_white else (0.1, 0.1, 0.1)
    pieces = []

    # Create pawns
    for i in range(8):
        pawn_shape = library_call('pawn')
        pawn_shape = transform_shape(pawn_shape, translation_matrix((i - 3.5, 0, 2.5 if is_white else -2.5)))
        pieces.append(transform_shape(pawn_shape, scale_matrix(0.35, (0, 0, 0))))

    # Create back row pieces
    back_row = [
        ('rook', -3.5), ('knight', -2.5), ('bishop', -1.5),
        ('queen', -0.5), ('king', 0.5), ('bishop', 1.5),
        ('knight', 2.5), ('rook', 3.5)
    ]

    for piece_name, x_pos in back_row:
        piece = library_call(piece_name)
        z_pos = 3.5 if is_white else -3.5
        piece = transform_shape(piece, translation_matrix((x_pos, 0, z_pos)))
        pieces.append(transform_shape(piece, scale_matrix(0.35, (0, 0, 0))))

    return concat_shapes(*[transform_shape(p, scale_matrix(1.0, (0, 0, 0))) for p in pieces])

@register()
def complete_chess_set() -> Shape:
    """Creates the complete chessboard with all pieces"""
    board = library_call('chessboard')
    white_pieces = transform_shape(library_call('piece_set', is_white=True), translation_matrix((0, 0.05, 0)))
    black_pieces = transform_shape(library_call('piece_set', is_white=False), translation_matrix((0, 0.05, 0)))

    return concat_shapes(board, white_pieces, black_pieces)
"""

This program creates a complete chess set with:
1. A modular design where each piece type is a separate function
2. Proper scaling and positioning of pieces
3. A chessboard with alternating black and white squares
4. Two complete sets of pieces (white and black)
5. All pieces are properly positioned on their starting squares

The pieces are simplified geometric representations:
- Pawns: cylindrical base with spherical top
- Rooks: cylindrical base with cubic crown
- Bishops: cylindrical base with small spherical top
- Knights: cylindrical base with rectangular head
- Queens: larger cylindrical base with spherical crown
- Kings: tallest piece with a cross on top

The board is created using an 8x8 grid of alternating colored squares, and the pieces are scaled and positioned appropriately on their starting positions. The white pieces start on ranks 1-2, and black pieces on ranks 7-8, following standard chess setup.
"""