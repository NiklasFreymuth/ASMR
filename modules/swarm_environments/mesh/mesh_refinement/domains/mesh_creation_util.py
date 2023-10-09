from typing import List, Tuple


def polygon_facets(start: int, end: int) -> List[Tuple[int, int]]:
    """
    Creates a list of facets/edges between the vertices of a polygon
    Args:
        start:
        end:

    Returns:

    """
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]
