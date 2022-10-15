import sys
sys.path.insert(1, '/home/che_h2/mzf0069/Documents/code/smolyak-surrogate-modeling')
import pytest
import numpy
from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator, IndexGrid,
                          generate_compositions)


def test_smolyak_initial_nested_basis_set_points():
    """Test initialized class returns correct points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class._points,
                          [0, -1, 1, -0.70710, 0.70710], atol=1e-10)


def test_smolyak_initial_nested_basis_set_functions():
    """Test initialized class return correct basis functions."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.3
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    test_class_basis_eval = []
    [test_class_basis_eval.append(basis.__call__(test_point))
     for basis in test_class._basis]

    assert numpy.allclose(test_class_basis_eval,
                          [1, 0.3, 2*(0.3**2)-1, 4*0.3**3-3*0.3,
                           8*0.3**4-8*0.3**2+1], atol=1e-10)


def test_ismolyak_nitial_nested_basis_set_levels():
    """Test initialized class returns correct levels."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(3)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_levels = [[0], [1, 2], [3, 4], [5, 6, 7, 8]]
    assert test_class._levels == expected_levels


def test_smolyak_grid_points_indexes_one_dimension():
    """Test integer grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0], [1], [2], [3], [4]]
    assert (test_class(1).grid_points_indexes
            == expected_integer_grid_points)


def test_smolyak_grid_points_indexes():
    """Test integer grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0, 0], [1, 0], [2, 0],
                                    [0, 1], [0, 2], [3, 0], [4, 0],
                                    [1, 1], [1, 2], [2, 1],
                                    [2, 2], [0, 3], [0, 4]]

    assert (test_class(2).grid_points_indexes
            == expected_integer_grid_points)


def test_smolyak_grid_points_one_dimension():
    """Test grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class(1).grid_points, [[0], [-1], [1],
           [-0.70710], [0.70710]], atol=1e-10)


def test_smolyak_grid_points():
    """Test grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_grid_points = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                            [-0.70710, 0], [0.70710, 0], [-1, -1],
                            [-1, 1], [1, -1], [1, 1], [0, -0.70710],
                            [0, 0.70710]]
    assert numpy.allclose(test_class(2).grid_points,
                          expected_grid_points, atol=1e-10)


def test_smolyak_grid_points_basis_one_dimension():
    """Test grid points' basis functions in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.3
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(1)._grid_points_basis
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
    test_point = 0.3
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(2)._grid_points_basis
    test_grid_points_basis_eval = []
    for grid_index in range(len(test_grid_points_basis)):
        test_grid_points_basis_eval.append([])
        for grid_basis_function in test_grid_points_basis[grid_index]:
            test_grid_points_basis_eval[grid_index].append(
                grid_basis_function.__call__(test_point))
    expected_grid_points_basis = [[1, 1], [0.3, 1], [2*(0.3**2)-1, 1],
                                  [1, 0.3], [1, 2*(0.3**2)-1]]

    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis, atol=1e-10)


def test_smolyak_indicies_one_dimension():
    """Test Smolyak indices in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_smolyak_indices = [[1], [2], [3]]

    assert test_class.indices(1) == expected_smolyak_indices


def test_smolyak_indices():
    """Test Smolyak indices."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_smolyak_indices = [[1, 1], [2, 1], [1, 2],
                                [3, 1], [2, 2], [1, 3]]

    assert test_class.indices(2) == expected_smolyak_indices


def test_smolyak_indices_expand():
    """Test the expand feature of smolyak_indices."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_smolyak_indices = [[1, 1], [2, 1], [1, 2],
                                [3, 1], [2, 2], [1, 3], [1, 4]]

    assert (test_class.indices(2, expand_indicy=[1, 4],
            drop_indicy=[]) == expected_smolyak_indices)


def test_smolyak_indices_drop():
    """Test the drop feature of smolyak_indices."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_smolyak_indices = [[1, 1], [2, 1], [1, 2],
                                [3, 1], [1, 3]]

    assert (test_class.indices(2, expand_indicy=[],
            drop_indicy=[2, 2]) == expected_smolyak_indices)


def test_smolyak_indicy_expand_drop():
    """Test the expand and the drop feature of smolyak_indices."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    expected_smolyak_indices = [[2, 1], [1, 2],
                                [3, 1], [2, 2], [1, 3], [1, 4]]

    assert (test_class.indices(2, expand_indicy=[1, 4],
            drop_indicy=[1, 1]) == expected_smolyak_indices)


def test_smolyak_indices_dimension_error():
    """Test if smolyak_indices raises an error when invalid indicy is given."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    with pytest.raises(IndexError):
        test_class.indices(2, expand_indicy=[1, 1, 1],
                           drop_indicy=[1, 2, 3])


def test_smolyak_indices_drop_error():
    """Test if drop_indicy raises an error when invalid indicy is given."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = SmolyakGridGenerator(test_nested_basis_set)
    with pytest.raises(ValueError):
        test_class.indices(2, drop_indicy=[1, 5])


def test_tensor_initial_basis_set_points():
    """Test initialized class returns correct points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = TensorGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class._points,
                          [0, -1, 1, -0.70710, 0.70710], atol=1e-10)


def test_tensor_initial_nested_basis_set_functions():
    """Test initialized class return correct basis functions."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.3
    test_class = TensorGridGenerator(test_nested_basis_set)
    test_class_basis_eval = []
    [test_class_basis_eval.append(basis.__call__(test_point))
     for basis in test_class._basis]

    assert numpy.allclose(test_class_basis_eval,
                          [1, 0.3, 2*(0.3**2)-1, 4*0.3**3-3*0.3,
                           8*0.3**4-8*0.3**2+1], atol=1e-10)


def test_tensor_grid_points_indexes_one_dimension():
    """Test integer grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0], [1], [2], [3], [4]]
    assert (test_class(1).grid_points_indexes
            == expected_integer_grid_points)


def test_tensor_grid_points_indexes():
    """Test integer grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_integer_grid_points = [[0, 0], [0, 1], [0, 2], [1, 0],
                                    [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    assert (test_class(2).grid_points_indexes
            == expected_integer_grid_points)


def test_tensor_grid_points_one_dimension():
    """Test grid points for one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_class = TensorGridGenerator(test_nested_basis_set)
    assert numpy.allclose(test_class(1).grid_points, [[0], [-1], [1],
           [-0.70710], [0.70710]], atol=1e-10)


def test_tensor_grid_points():
    """Test grid points."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_class = TensorGridGenerator(test_nested_basis_set)
    expected_grid_points = [[0, 0], [0, -1], [0, 1], [-1, 0],
                            [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]]
    assert numpy.allclose(test_class(2).grid_points,
                          expected_grid_points, atol=1e-10)


def test_tensor_grid_points_basis_one_dimesnion():
    """Test grid points' basis functions in one dimension."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(2)
    test_point = 0.3
    test_class = TensorGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(1)._grid_points_basis
    test_grid_points_basis_eval = []
    for grid in test_grid_points_basis:
        for grid_point_basis in grid:
            test_grid_points_basis_eval.append(
                grid_point_basis.__call__(test_point))
    expected_grid_points_basis_eval = [1, 0.3, 2*(0.3**2)-1, 4*0.3**3-3*0.3,
                                       8*0.3**4-8*0.3**2+1]
    assert numpy.allclose(test_grid_points_basis_eval,
                          expected_grid_points_basis_eval, atol=1e-10)


def test_tensor_grid_points_basis():
    """Test grid points' basis functions."""
    test_nested_basis_set = ChebyshevFirstKind.make_nested_set(1)
    test_point = 0.3
    test_class = TensorGridGenerator(test_nested_basis_set)
    test_grid_points_basis = test_class(2)._grid_points_basis
    test_grid_points_basis_eval = []
    for grid_index in range(len(test_grid_points_basis)):
        test_grid_points_basis_eval.append([])
        for grid_basis_function in test_grid_points_basis[grid_index]:
            test_grid_points_basis_eval[grid_index].append(
                grid_basis_function.__call__(test_point))
    expected_grid_points_basis = [[1, 1], [1, 0.3], [1, 2*(0.3**2)-1],
                                  [0.3, 1], [0.3, 0.3], [0.3, 2*(0.3**2)-1],
                                  [2*(0.3**2)-1, 1], [2*(0.3**2)-1, 0.3],
                                  [2*(0.3**2)-1, 2*(0.3**2)-1]]

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
