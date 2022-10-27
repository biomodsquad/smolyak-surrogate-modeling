import numpy
import pytest

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator,
                          IndexGrid, NestedIndexGrid, generate_compositions)


def test_error_basis_set_type():
    """Test if raise TypeError works for IndexGridGenerator."""
    basis_set = "not a BasisFunctionSet"
    with pytest.raises(TypeError):
        IndexGridGenerator(basis_set)


def test_error_nested_basis_set_type():
    """Test if raise TypeError works for SmolyakGridGenerator."""
    basis_set = "not a NestedBasisFunctionSet"
    with pytest.raises(TypeError):
        SmolyakGridGenerator(basis_set)


def test_smolyak_grid_points_indexes_one_dimension():
    """Test integer grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0], [1], [2], [3], [4]]
    assert (test_class(1).indexes
            == expected_integer_grid_points)


def test_smolyak_grid_points_indexes():
    """Test integer grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0, 0], [1, 0], [2, 0],
                                    [0, 1], [0, 2], [3, 0], [4, 0],
                                    [1, 1], [1, 2], [2, 1],
                                    [2, 2], [0, 3], [0, 4]]
    assert (test_class(2).indexes
            == expected_integer_grid_points)


def test_smolyak_levels_one_dimension():
    """Test levels of the grid points. in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(3)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_levels = [[0], [1], [1], [2], [2], [3], [3], [3], [3]]
    assert test_class(1).levels == expected_levels


def test_smolyak_levels():
    """Test levels of the grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_levels = [[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [2, 0],
                       [2, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 2], [0, 2]]
    assert test_class(2).levels == expected_levels


def test_smolyak_grid_points_one_dimension():
    """Test grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class(1).points, [[0], [-1], [1],
           [-0.70710], [0.70710]], atol=1e-10)


def test_smolyak_grid_points():
    """Test grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_grid_points = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                            [-0.70710, 0], [0.70710, 0], [-1, -1],
                            [-1, 1], [1, -1], [1, 1], [0, -0.70710],
                            [0, 0.70710]]
    assert numpy.allclose(test_class(2).points,
                          expected_grid_points, atol=1e-10)


def test_smolyak_grid_points_basis_one_dimension():
    """Test grid points' basis functions in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.3
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(1).basis
    test_grid_points_basis_eval = []
    for grid in test_grid_points_basis:
        for grid_point_basis in grid:
            test_grid_points_basis_eval.append(
                grid_point_basis.__call__(test_point))
    expected_grid_points_basis_eval = [1, 0.3, 2*(0.3**2)-1, 4*0.3**3-3*0.3,
                                       8*0.3**4-8*0.3**2+1]
    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis_eval, atol=1e-10)


def test_smolyak_grid_points_basis():
    """Test grid points' basis functions."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_point = 0.72
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(2).basis
    test_grid_points_basis_eval = []
    for grid_index in range(len(test_grid_points_basis)):
        test_grid_points_basis_eval.append([])
        for grid_basis_function in test_grid_points_basis[grid_index]:
            test_grid_points_basis_eval[grid_index].append(
                grid_basis_function.__call__(test_point))
    expected_grid_points_basis = [[1, 1], [0.72, 1], [2*(0.72**2)-1, 1],
                                  [1, 0.72], [1, 2*(0.72**2)-1]]

    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis, atol=1e-10)


def test_tensor_grid_points_indexes_one_dimension():
    """Test integer grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0], [1], [2], [3], [4]]
    assert (test_class(1).indexes
            == expected_integer_grid_points)


def test_tensor_grid_points_indexes():
    """Test integer grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0, 0], [0, 1], [0, 2], [1, 0],
                                    [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    assert (test_class(2).indexes
            == expected_integer_grid_points)


def test_tensor_grid_points_one_dimension():
    """Test grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = TensorGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class(1).points, [[0], [-1], [1],
           [-0.70710], [0.70710]], atol=1e-10)


def test_tensor_grid_points():
    """Test grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_grid_points = [[0, 0], [0, -1], [0, 1], [-1, 0],
                            [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]]
    assert numpy.allclose(test_class(2).points,
                          expected_grid_points, atol=1e-10)


def test_tensor_grid_points_basis_one_dimesnion():
    """Test grid points' basis functions in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.39
    test_class = TensorGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(1).basis
    test_grid_points_basis_eval = []
    for grid in test_grid_points_basis:
        for grid_point_basis in grid:
            test_grid_points_basis_eval.append(
                grid_point_basis.__call__(test_point))
    expected_grid_points_basis_eval = [1, 0.39, 2*(0.39**2)-1,
                                       4*0.39**3-3*0.39,
                                       8*0.39**4-8*0.39**2+1]
    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis_eval, atol=1e-10)


def test_tensor_grid_points_basis():
    """Test grid points' basis functions."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_point = 0.5
    test_class = TensorGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(2).basis
    test_grid_points_basis_eval = []
    for grid_index in range(len(test_grid_points_basis)):
        test_grid_points_basis_eval.append([])
        for grid_basis_function in test_grid_points_basis[grid_index]:
            test_grid_points_basis_eval[grid_index].append(
                grid_basis_function.__call__(test_point))
    expected_grid_points_basis = [[1, 1], [1, 0.5], [1, 2*(0.5**2)-1],
                                  [0.5, 1], [0.5, 0.5], [0.5, 2*(0.5**2)-1],
                                  [2*(0.5**2)-1, 1], [2*(0.5**2)-1, 0.5],
                                  [2*(0.5**2)-1, 2*(0.5**2)-1]]

    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis, atol=1e-10)


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
