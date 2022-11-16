import numpy
import pytest

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator, IndexGrid
from smolyay.surrogate import Surrogate


def function_0(x):
    """Test function 0."""
    x1, x2, x3 = x
    return 2*x1 + x2 - x3


def function_0_shifted(x):
    """Test fucntion 0 which is shifted."""
    x1, x2, x3 = x
    return 2*(x1 - 1) + (x2 + 1) - x3


def function_1(x):
    """Test fucntion 1."""
    x1, x2 = x
    return x1 + (2*x2**2 - 1)


def function_2(x):
    """Test funciton 2."""
    return x**2 - 3 * (2 + x) - x


def branin(x):
    """Branin function."""
    x1, x2 = x
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
        branin_at_grids.append(branin(i))
    return branin_at_grids


def test_initialization_1d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2))
    domain = (-1, 1)
    surrogate_object = Surrogate(domain, grid_generator)
    assert numpy.allclose(surrogate_object.domain,
                          numpy.array(domain, ndmin=2))
    assert surrogate_object._coefficients is None
    assert surrogate_object._grid is None
    grid_object = surrogate_object.grid
    assert isinstance(grid_object, IndexGrid)
    assert surrogate_object.dimension == 1


def test_initialization_2d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2))
    domain = [(-5, 5), (-8, 0)]
    grid_generator_ = 'not an IndexGridGenerator'
    surrogate_object = Surrogate(domain, grid_generator)
    assert numpy.allclose(surrogate_object.domain,
                          numpy.array(domain, ndmin=2))
    with pytest.raises(TypeError):
        Surrogate(domain, grid_generator_)
    assert surrogate_object._coefficients is None
    grid_object = surrogate_object.grid
    assert isinstance(grid_object, IndexGrid)
    assert surrogate_object.dimension == 2


def test_map_function_1d():
    """Test if points are transfromed properly."""
    grid = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(1))
    domain = (-8, 12)
    new_domain = (0, 1)
    f = Surrogate(domain, grid)
    assert f._mapdomain(2, domain, new_domain) == pytest.approx(0.5)
    assert numpy.allclose(f._mapdomain([-3, 7],
                                       domain, new_domain), [0.25, 0.75])


def test_map_function_2d():
    """Test if points are transfromed properly."""
    grid = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(1))
    domain = ((-10, 10), (0, 2))
    new_domain = ((0, 1), (-1, 1))
    f = Surrogate(domain, grid)
    assert numpy.allclose(f._mapdomain((-10, 0), domain, new_domain), [0, -1])
    assert numpy.allclose(f._mapdomain([(0, 1), (10, 2)],
                                       domain, new_domain), [[0.5, 0], [1, 1]])


def test_basis_matrix():
    """Test if a correct basis matrix is generated."""
    grid_generator = SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(1))
    surrogate_object = Surrogate([(-7, 3), (-18, 9)], grid_generator)
    expected_basis_matrix = numpy.array(
        [[1, 0, -1, 0, -1], [1, -1, 1, 0, -1], [1, 1, 1, 0, -1],
         [1, 0, -1, -1, 1], [1, 0, -1, 1, 1]])
    assert numpy.allclose(expected_basis_matrix,
                          surrogate_object._make_basis_matrix())


def test_surrogate_0():
    """Test if surrogate generates exact results for test fucntion 0."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1), (-1, 1)],
                                 SmolyakGridGenerator(
                                     ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_0, 'lu')
    test_point = (0.649, 0, -0.9)
    test_point_ = (-0.885, 1, 0.275)
    assert numpy.allclose(surrogate_object(test_point),
                          function_0([0.649, 0, -0.9]))
    assert numpy.allclose(surrogate_object(test_point_),
                          function_0([-0.885, 1, 0.275]))


def test_surrogate_1():
    """Test if surrogate generates exact results for test for Chebyshevs."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1)], SmolyakGridGenerator(
        ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_1, 'lstsq')
    test_point = (0.649, -0.9)
    test_point_ = (-0.885, 1)
    assert numpy.allclose(surrogate_object(test_point),
                          function_1([0.649, -0.9]))
    assert numpy.allclose(surrogate_object(test_point_),
                          function_1([-0.885, 1]))


def test_surrogate_0_shifted():
    """Test if surrogate generates exact results for a shifted function 0."""
    surrogate_object = Surrogate([(-1, 1), (-1, 1), (-1, 1)],
                                 SmolyakGridGenerator(
                                     ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object_ = Surrogate([(0, 2), (-2, 0), (-1, 1)],
                                  SmolyakGridGenerator(
                                      ChebyshevFirstKind.make_nested_set(2)))
    surrogate_object.train(function_0, 'lu')
    surrogate_object_.train(function_0_shifted, 'lu')
    assert numpy.allclose(surrogate_object.coefficients,
                          surrogate_object_.coefficients)


def test_surrogate_2():
    """Test if surrogate generates exact results for simple 1D function."""
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
                          branin([8, 0.75]), atol=1e-10)
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
