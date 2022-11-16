import numpy
import pytest

from smolyay.basis import (ChebyshevFirstKind, BasisFunctionSet)
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator, generate_compositions)


def test_error_basis_set_type():
    """Test if raise TypeError works for IndexGridGenerator."""
    basis_set = "not a BasisFunctionSet"
    with pytest.raises(TypeError):
        IndexGridGenerator(basis_set)

def test_error_nested_basis_set_type():
    """Test if raise TypeError works for SmolyakGridGenerator."""
    nested_set_object = ChebyshevFirstKind.make_nested_set(3)
    points = nested_set_object.points
    basis_functions = nested_set_object.basis_functions
    basis_set = BasisFunctionSet(points, basis_functions)
    with pytest.raises(TypeError):
        SmolyakGridGenerator(basis_set)

def test_smolyak_grid_1d():
    """Test grids generated by Smolyak method."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    grid = grid_gen(1)
    assert grid.indexes == [0, 1, 2, 3, 4]
    assert grid.levels == [0, 1, 1, 2, 2]
    assert numpy.allclose(grid.points, [0, -1, 1, -0.70710, 0.70710], atol=1e-10)

def test_smolyak_grid_2d():
    """Test grids generated by Smolyak method."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0], [1, 0], [2, 0],
                            [0, 1], [0, 2], [3, 0],
                            [4, 0],
                            [1, 1], [1, 2], [2, 1],
                            [2, 2], [0, 3], [0, 4]]
    assert grid.levels == [[0, 0], [1, 0], [1, 0], [0, 1], [0, 1],
                           [2, 0], [2, 0], [1, 1], [1, 1], [1, 1],
                           [1, 1], [0, 2], [0, 2]]
    assert numpy.allclose(grid.points,
                          [[0, 0], [-1, 0], [1, 0], [0, -1],
                           [0, 1],
                           [-0.70710, 0], [0.70710, 0],
                           [-1, -1],
                           [-1, 1], [1, -1], [1, 1],
                           [0, -0.70710],
                           [0, 0.70710]],
                          atol=1e-10)

def test_tensor_grid_1d():
    grid_gen = TensorGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    grid = grid_gen(1)
    assert grid.indexes == [0, 1, 2, 3, 4]
    assert numpy.allclose(grid.points, [0, -1, 1, -0.70710, 0.70710], atol=1e-10)

def test_tensor_grid_2d():
    """Test grids generated by Tensor product."""
    grid_gen = TensorGridGenerator(ChebyshevFirstKind.make_nested_set(1))
    grid = grid_gen(2)
    assert grid.indexes == [[0, 0], [0, 1], [0, 2],
                            [1, 0],
                            [1, 1], [1, 2], [2, 0],
                            [2, 1], [2, 2]]
    assert numpy.allclose(
        grid.points,
        [[0, 0], [0, -1], [0, 1], [-1, 0],
         [-1, -1], [-1, 1], [1, 0],
         [1, -1], [1, 1]],
        atol=1e-10)

def test_generate_compositions_include_zero_true():
    """Test the generate compositions function if include_zero is true."""
    composition_expected = [[6, 0], [5, 1], [4, 2], [3, 3],
                            [2, 4], [1, 5], [0, 6]]
    composition_obtained = []
    composition_obtained = list(generate_compositions(6, 2, include_zero=True))
    assert composition_obtained == composition_expected


def test_generate_compositions_include_zero_false():
    """Test the generate compositions function if include_zero is false."""
    composition_expected = [[5, 1], [4, 2], [3, 3], [2, 4], [1, 5]]
    composition_obtained = list(generate_compositions(6,
                                                      2, include_zero=False))
    assert composition_obtained == composition_expected


def test_generate_compositions_zero_false_error():
    """Test that generate compositions raises an error for invalid input."""
    with pytest.raises(ValueError):
        list(generate_compositions(6, 7, include_zero=False))
