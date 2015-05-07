"""
Split a rectangle into n panes.
"""
from __future__ import division

__all__ = ["max_tile_size"]

import math


def max_tile_size(tile_count, rect_size):
    """
    Determine the maximum sized tile possible.

    Keyword arguments:
    tile_count -- Number of tiles to fit
    rect_size -- 2-tuple of rectangle size as (width, height)
    """

    # If the rectangle is taller than it is wide, reverse its dimensions
    if rect_size[0] < rect_size[1]:
        rect_size = rect_size[1], rect_size[0]

    # Rectangle aspect ratio
    rect_ar = rect_size[0] / rect_size[1]

    # tiles_max_height is the square root of tile_count, rounded up
    tiles_max_height = int(math.ceil(math.sqrt(tile_count)))

    best_tile_size = 0

    # i in the range [1, tile_max_height], inclusive
    for i in range(1, tiles_max_height + 1):

        # tiles_used is the arrangement of tiles (width, height)
        tiles_used = math.ceil(tile_count / i), i

        # tiles_ar is the aspect ratio of this arrangement
        tile_ar = tiles_used[0] / tiles_used[1]

        # Calculate the size of each tile
        # Tile pattern is flatter than rectangle
        if tile_ar > rect_ar:
            tile_size = rect_size[0] / tiles_used[0]
        # Tile pattern is skinnier than rectangle
        else:
            tile_size = rect_size[1] / tiles_used[1]

        # Check if this is the best answer so far
        if tile_size > best_tile_size:
            best_tile_size = tile_size

    return best_tile_size

# print(max_tile_size(6, (100, 100)))
