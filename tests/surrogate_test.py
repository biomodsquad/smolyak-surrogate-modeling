import numpy
import pytest

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator, IndexGrid
from smolyay.surrogate import Surrogate


def function_0(x1, x2, x3):
    """Test function 0."""
    return 2*x1 + x2 - x3


def function_0_shifted(x1, x2, x3):
    """Test fucntion 0 which is shifted."""
    return 2*(x1 - 1) + (x2 + 1) - x3


def function_1(x1, x2):
    """Test fucntion 1."""
    return x1 + (2*x2**2 - 1)


def function_2(x1):
    """Test funciton 2."""
    return x1**2 - 3 * (2 + x1) - x1


def branin(x1, x2):
    """Branin function."""
    branin1 = (x2 - 5.1 * x1 ** (2)/(4 * numpy.pi ** 2)
               + 5 * x1 / (numpy.pi) - 6) ** 2
    branin2 = 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(x1)
    branin3 = 10
    branin_function = branin1 + branin2 + branin3
    return branin_function


def branin_at_grids():
    """Evaluate Branin function at grid points."""
    points = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4)
                                  )(2).points
    transformed_points = numpy.zeros((len(points), 2))
    transformed_points[:, 0] = numpy.polynomial.polyutils.mapdomain(
                            numpy.array(points)[:, 0],
                            (-1, 1), (-5, 10))
    transformed_points[:, 1] = numpy.polynomial.polyutils.mapdomain(
                            numpy.array(points)[:, 1],
                            (-1, 1), (0, 15))
    branin_at_grids = []
    for i in transformed_points:
        branin_at_grids.append(branin(i[0], i[1]))
    return branin_at_grids


def test_initialization():
    """Test if class is properly intiallized."""
    test_grid_generator = SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2))
    test_domain_one_dimension = (-1, 1)
    test_domain = [(-5, 5), (-8, 0)]
    test_error_generator_type = 'not an IndexGridGenerator'
    surrogate_object_one_dimension = Surrogate(
        test_domain_one_dimension, test_grid_generator)
    surrogate_object = Surrogate(test_domain, test_grid_generator)
    grid_object = surrogate_object.grid

    assert numpy.allclose(surrogate_object_one_dimension.domain,
                          numpy.array(test_domain_one_dimension, ndmin=2))
    assert numpy.allclose(surrogate_object.domain,
                          numpy.array(test_domain, ndmin=2))
    with pytest.raises(TypeError):
        Surrogate(test_domain, test_error_generator_type)
    assert surrogate_object._coefficients is None
    assert surrogate_object_one_dimension._grid is None
    assert isinstance(grid_object, IndexGrid)


def test_basis_matrix():
    """Test if a correct basis matrix is generated."""
    test_grid_generator = SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(1))
    test_class = Surrogate([(-7, 3), (-18, 9)], test_grid_generator)
    expected_basis_matrix = numpy.array(
        [[1, 0, -1, 0, -1], [1, -1, 1, 0, -1], [1, 1, 1, 0, -1],
         [1, 0, -1, -1, 1], [1, 0, -1, 1, 1]])
    assert numpy.allclose(expected_basis_matrix,
                          test_class._make_basis_matrix())


def test_surrogate_0():
    """Test if surrogate generates exact result for test fucntion 0."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1), (-1, 1)],
                                 SmolyakGridGenerator(
                                     ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_0, 'lu')
    test_point = (0.649, 0, -0.9)
    test_point_ = (-0.885, 1, 0.275)
    assert numpy.allclose(surrogate_object(test_point),
                          function_0(0.649, 0, -0.9))
    assert numpy.allclose(surrogate_object(test_point_),
                          function_0(-0.885, 1, 0.275))


def test_surrogate_1():
    """Test if surrogate generate exact result for test for Chebyshevs."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_1, 'lstsq')
    test_point = (0.649, -0.9)
    test_point_ = (-0.885, 1)
    assert numpy.allclose(surrogate_object(test_point),
                          function_1(0.649, -0.9))
    assert numpy.allclose(surrogate_object(test_point_),
                          function_1(-0.885, 1))


def test_surrogate_0_shifted():
    """Test if surrogate generates exact result a shifted test function."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1), (-1, 1)],
                                 SmolyakGridGenerator(
                                     ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object_ = Surrogate([(0, 2), (-2, 0), (-1, 1)],
                                  SmolyakGridGenerator(
                                      ChebyshevFirstKind.make_nested_set(2)))
    assert numpy.allclose(surrogate_object.train(function_0, 'lu'),
                          surrogate_object_.train(function_0_shifted, 'lu'),
                          atol=1e10)


def test_surrogate_function_2():
    """Test of surrogate can be built for 1D test function."""
    domain = (-10, 10)
    surrogate_object = Surrogate(domain, SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_2, 'inv')
    test_point = 0.367
    assert numpy.allclose(surrogate_object(test_point), function_2(test_point))


def test_train_from_data():
    """Test if train_from_data_works."""
    surrogate_object = Surrogate([(-5, 10), (0, 15)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(4)))
    surrogate_object_ = Surrogate([(-5, 10), (0, 15)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(4)))
    surrogate_object.train_from_data(branin_at_grids(), 'lu')
    assert numpy.allclose(surrogate_object((8, 0.75)),
                          branin(8, 0.75), atol=1e-10)
    with pytest.raises(IndexError):
        surrogate_object_.train_from_data(branin_at_grids()[:5], 'lu')


def test_error_function_needs_training():
    """Test if surrogate is generated without training."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2)))
    test_point = (0.649, 0)
    with pytest.raises(ValueError):
        surrogate_object(test_point)


def test_error_input_surrogate():
    """Test the error related to surrogate's input dimension."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_1, 'lstsq')
    test_point = (0.649, 0, 0.5)
    with pytest.raises(IndexError):
        surrogate_object(test_point)
