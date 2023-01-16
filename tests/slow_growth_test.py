import pytest

import numpy

from smolyay.basis import (ChebyshevFirstKind, NestedBasisFunctionSet)
from smolyay.grid import SmolyakGridGenerator
from smolyay.surrogate import Surrogate


def make_slow_nested_set(exactness):
    """Create a slow growth nested set of Chebyshev polynomials.

    A nested set is created up to a given level of ``exactness``,
    which corresponds to a highest-order Chebyshev polynomial of
    degree ``n = 2**exactness``.

    Each nesting level corresponds to the increasing powers of 2 going up to
    ``2**exactness``, with the first level being a special case. The 
    generating Chebyshev polynomials are hence of degree (0, 2, 4, ...).
    Each new point added in a level is paired with a basis function of 
    increasing order. However, if the sum of the current points exceed a
    precision rule, then no points would be added.

    The precision rule is defined as 2 * ``exactness`` + 1. For a given
    ``exactness``, points are added at levels where the precision rule
    is greater than the sum of all points added at previous levels.

    For example, for an ``exactness`` of 3, the generating polynomials are
    of degree 0, 2, 4, and 8, at each of 4 levels. There are 1, 2, 2, and 4
    new points added at each level. The polynomial at level 0 is of degree 
    0, the polynomials at level 1 are of degrees 1 and 2, those at level 2 
    are of degree 3 and 4, and those at level 3 are of degrees 5, 6, 7, and
    8.

    For an ``exactness`` of 4, there would be 1, 2, 2, 4, and 8 new points
    added at each level. However, the precision rule states that for level
    4 the precision requirement is 9. The precision before points are
    added is 1 + 2 + 2 + 4 = 9, meaning that instead of adding 8 points to
    the new level, 0 are added.

    Parameters
    ----------
    exactness : int
        Level of exactness.

    Returns
    -------
    NestedBasisFunctionSet
        Nested Chebyshev polynomials of the first kind.

    """
    basis_functions = []
    levels = []
    points = []
    rule_add = -1
    precision_has = 0
    for i in range(0, exactness+1):
        if 2 * i + 1 > precision_has:
            rule_add = rule_add + 1
            if rule_add > 1:
                start_level = 2**(rule_add-1)+1
                end_level = 2**rule_add
            elif rule_add == 1:
                start_level = 1
                end_level = 2
            else:
                start_level = 0
                end_level = 0
            level_range = range(start_level, end_level+1)
            precision_has = precision_has + len(level_range)
            levels.append(list(level_range))

            basis_functions.extend(ChebyshevFirstKind(n) for n in level_range)
            for p in basis_functions[end_level].points:
                if not numpy.isclose(points, p).any():
                    points.append(p)
        else:
            levels.append([])
    return NestedBasisFunctionSet(points,basis_functions,levels)

def test_slow_growth_small():
    """Test grid indexes if levels property is empty"""
    levels = [[0], [1,2], []]
    points = [0, -1, 1]
    basis_functions = [ChebyshevFirstKind(n) for n in range(3)]
    nest_set = NestedBasisFunctionSet(points,basis_functions,levels)
    grid_gen = SmolyakGridGenerator(nest_set)
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0],[1, 0],[2, 0],[0, 1],[0, 2],[1, 1],[1, 2],
            [2, 1],[2, 2]]

def test_slow_growth_exactness_4():
    """Test grid indexes expected of slow growth exactness 4"""
    slow_nested_set = make_slow_nested_set(4)
    grid_gen = SmolyakGridGenerator(slow_nested_set)
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0],[1, 0],[2, 0],[0, 1],[0, 2],[3, 0],[4, 0],
                            [1, 1],[1, 2],[2, 1],[2, 2],[0, 3],[0, 4],[5, 0],
                            [6, 0],[7, 0],[8, 0],[3, 1],[3, 2],[4, 1],[4, 2],
                            [1, 3],[1, 4],[2, 3],[2, 4],[0, 5],[0, 6],[0, 7],
                            [0, 8],[5, 1],[5, 2],[6, 1],[6, 2],[7, 1],[7, 2],
                            [8, 1],[8, 2],[3, 3],[3, 4],[4, 3],[4, 4],[1, 5],
                            [1, 6],[1, 7],[1, 8],[2, 5],[2, 6],[2, 7],[2, 8]]
    
def test_slow_growth_exactness_5():
    """Test grid indexes expected of slow growth exactness 5"""
    slow_nested_set = make_slow_nested_set(5)
    grid_gen = SmolyakGridGenerator(slow_nested_set)
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0],[1, 0],[2, 0],[0, 1],[0, 2],[3, 0],[4, 0],
                            [1, 1],[1, 2],[2, 1],[2, 2],[0, 3],[0, 4],[5, 0],
                            [6, 0],[7, 0],[8, 0],[3, 1],[3, 2],[4, 1],[4, 2],
                            [1, 3],[1, 4],[2, 3],[2, 4],[0, 5],[0, 6],[0, 7],
                            [0, 8],[5, 1],[5, 2],[6, 1],[6, 2],[7, 1],[7, 2],
                            [8, 1],[8, 2],[3, 3],[3, 4],[4, 3],[4, 4],[1, 5],
                            [1, 6],[1, 7],[1, 8],[2, 5],[2, 6],[2, 7],[2, 8],
                            [9, 0],[10, 0],[11, 0],[12, 0],[13, 0],[14, 0],
                            [15, 0],[16, 0],[5, 3],[5, 4],[6, 3],[6, 4],[7, 3],
                            [7, 4],[8, 3],[8, 4],[3, 5],[3, 6],[3, 7],[3, 8],
                            [4, 5],[4, 6],[4, 7],[4, 8],[0, 9],[0, 10],[0, 11],
                            [0, 12],[0, 13],[0, 14],[0, 15],[0, 16]]

