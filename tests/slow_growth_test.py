import pytest

from smolyay.basis import (ChebyshevFirstKind, NestedBasisFunctionSet)
from smolyay.grid import SmolyakGridGenerator

def test_slow_growth():
    """Test grid indexes if levels property is empty"""
    levels = [[0], [1,2], []]
    points = [0, -1, 1]
    basis_functions = [ChebyshevFirstKind(n) for n in range(3)]
    nest_set = NestedBasisFunctionSet(points,basis_functions,levels)
    grid_gen = SmolyakGridGenerator(nest_set)
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0],[1, 0],[2, 0],[0, 1],[0, 2],[1, 1],[1, 2],
            [2, 1],[2, 2]]
