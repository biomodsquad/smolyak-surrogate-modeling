import numpy
import pytest

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator, IndexGrid
from smolyay.normalize import (
    Normalizer,
    NullNormalizer,
    SymmetricalLogNormalizer,
    IntervalNormalizer,
    ZScoreNormalizer,
)
from smolyay.surrogate import Surrogate, GradientSurrogate


def function_0(x):
    """Test function 0."""
    x1, x2, x3 = x
    return 2 * x1 + x2 - x3


def function_0_shifted(x):
    """Test fucntion 0 which is shifted."""
    x1, x2, x3 = x
    return 2 * (x1 - 1) + (x2 + 1) - x3


def function_1(x):
    """Test fucntion 1."""
    x1, x2 = x
    return x1 + (2 * x2**2 - 1)


def function_2(x):
    """Test funciton 2."""
    return x**2 - 3 * (2 + x) - x


def function_3(x):
    """Test function 3 (gradient)."""
    # function f = x1*x2 - 2*x2
    x1, x2 = x
    return x2, x1 - 2


def function_4(x):
    """Test function 4 (gradient)."""
    # function f = x**3 -2*x
    return 3 * x**2 - 2


def branin(x):
    """Branin function."""
    x1, x2 = x
    branin1 = (x2 - 5.1 * x1 ** (2) / (4 * numpy.pi**2) + 5 * x1 / (numpy.pi) - 6) ** 2
    branin2 = 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(x1)
    branin3 = 10
    branin_function = branin1 + branin2 + branin3
    return branin_function


def test_initialization_1d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    domain = (-1, 1)
    surrogate = Surrogate(domain, grid_generator)
    assert numpy.allclose(surrogate.domain, domain)
    assert surrogate.coefficients is None
    assert isinstance(surrogate.grid, IndexGrid)
    assert surrogate.dimension == 1
    assert numpy.allclose(surrogate.points, [0, -1, 1, -0.70710, 0.70710])


def test_gradient_initialization_1d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    domain = (-1, 1)
    surrogate = GradientSurrogate(domain, grid_generator)
    assert numpy.allclose(surrogate.domain, domain)
    assert surrogate.coefficients is None
    assert isinstance(surrogate.grid, IndexGrid)
    assert surrogate.dimension == 1
    assert numpy.allclose(surrogate.points, [0, -1, 1, -0.70710, 0.70710])


def test_initialization_2d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    domain = [(-5, 5), (-8, 0)]
    surrogate = Surrogate(domain, grid_generator)
    assert numpy.allclose(surrogate.domain, domain)
    assert surrogate.coefficients is None
    assert isinstance(surrogate.grid, IndexGrid)
    assert surrogate.dimension == 2
    assert numpy.allclose(
        surrogate.points,
        [
            [0, -4],
            [-5, -4],
            [5, -4],
            [0, -8],
            [0, 0],
            [-3.5355, -4],
            [3.5355, -4],
            [-5, -8],
            [-5, 0],
            [5, -8],
            [5, 0],
            [0, -6.8284],
            [0, -1.1715],
        ],
        atol=1e-4,
    )
    assert isinstance(surrogate.norm, NullNormalizer)


def test_gradient_initialization_2d():
    """Test if class is properly intiallized."""
    grid_generator = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    domain = [(-5, 5), (-8, 0)]
    surrogate = GradientSurrogate(domain, grid_generator)
    assert numpy.allclose(surrogate.domain, domain)
    assert surrogate.coefficients is None
    assert isinstance(surrogate.grid, IndexGrid)
    assert surrogate.dimension == 2
    assert numpy.allclose(
        surrogate.points,
        [
            [0, -4],
            [-5, -4],
            [5, -4],
            [0, -8],
            [0, 0],
            [-3.5355, -4],
            [3.5355, -4],
            [-5, -8],
            [-5, 0],
            [5, -8],
            [5, 0],
            [0, -6.8284],
            [0, -1.1715],
        ],
        atol=1e-4,
    )
    assert isinstance(surrogate.norm, NullNormalizer)


def test_error_grid_generator_type():
    """Test grid generator's type."""
    domain = [(-5, 5), (-8, 0)]
    with pytest.raises(TypeError):
        Surrogate(domain, "not a grid generator")


def test_error_gradient_grid_generator_type():
    """Test grid generator's type."""
    domain = [(-5, 5), (-8, 0)]
    with pytest.raises(TypeError):
        GradientSurrogate(domain, "not a grid generator")


def test_map_function_1d():
    """Test if points are transfromed properly."""
    grid = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(1))
    domain = (-8, 12)
    new_domain = (0, 1)
    f = Surrogate(domain, grid)
    assert f._mapdomain(2, domain, new_domain)
    assert numpy.allclose(f._mapdomain([-3, 7], domain, new_domain), [0.25, 0.75])


def test_map_function_2d():
    """Test if points are transfromed properly."""
    grid = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(1))
    domain = ((-10, 10), (0, 2))
    new_domain = ((0, 1), (-1, 1))
    f = Surrogate(domain, grid)
    assert numpy.allclose(f._mapdomain((-10, 0), domain, new_domain), [0, -1])
    assert numpy.allclose(
        f._mapdomain([(0, 1), (10, 2)], domain, new_domain), [[0.5, 0], [1, 1]]
    )


@pytest.mark.parametrize("linear_solver", ["lu", "lstsq", "inv"])
def test_surrogate_0(linear_solver):
    """Test if surrogate generates exact results for test fucntion 0."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1), (-1, 1)], grid_gen)
    surrogate.train(function_0, linear_solver)
    # random points in the domain
    points = [(0.649, 0, -0.9), (-0.885, 1, 0.275)]
    surrogate_values = [surrogate(x) for x in points]
    exact_values = [function_0(x) for x in points]
    assert numpy.allclose(surrogate_values, exact_values)


def test_surrogate_1():
    """Test if surrogate generates exact results for test for Chebyshevs."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen)
    surrogate.train(function_1)
    # random points in the domain
    points = [(0.649, -0.9), (-0.885, 1)]
    surrogate_values = [surrogate(x) for x in points]
    exact_values = [function_1(x) for x in points]
    assert numpy.allclose(surrogate_values, exact_values)


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_surrogate_1_multi_input(norm):
    """Test if surrogate generates exact results when call has >1 input."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen, norm)
    surrogate.train(function_1)
    # random points in the domain
    numpy.random.seed(567)
    points_1 = [(0.649, -0.9), (-0.885, 1)]
    points_2 = numpy.random.rand(3, 4, 2) * 2 - 1
    exact_values_1 = [function_1(x) for x in points_1]
    exact_values_2 = numpy.zeros(points_2.shape[:-1])
    for i in range(points_2.shape[0]):
        for j in range(points_2.shape[1]):
            exact_values_2[i, j] = function_1(tuple(points_2[i, j]))
    surrogate_values_1 = surrogate(points_1)
    surrogate_values_2 = surrogate(points_2)
    assert numpy.allclose(exact_values_1, surrogate_values_1)
    assert numpy.allclose(exact_values_2, surrogate_values_2)


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_surrogate_2D_multi_shape(norm):
    """Test if surrogate call generates the correct shape for 2 dimensions."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen, norm)
    surrogate.train(function_1)
    # arrays of various shapes with points in the domain
    assert numpy.isscalar(surrogate([1, 1]))
    assert surrogate([[1, 1]]).shape == (1,)
    assert surrogate([[1, 1], [0.5, 0.5]]).shape == (2,)
    assert surrogate(numpy.ones((1, 1, 1, 2))).shape == (1, 1, 1)
    assert surrogate(numpy.ones((4, 3, 2, 2))).shape == (4, 3, 2)
    assert surrogate(numpy.ones((4, 3, 5, 2))).shape == (4, 3, 5)


def test_surrogate_0_shifted():
    """Test if surrogate generates exact results for a shifted function 0."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate_1 = Surrogate([(-1, 1), (-1, 1), (-1, 1)], grid_gen)
    surrogate_2 = Surrogate([(0, 2), (-2, 0), (-1, 1)], grid_gen)
    surrogate_1.train(function_0)
    surrogate_2.train_from_data(surrogate_1.data)
    point, point_shifted = (0, 0.5, -0.5), (1, -0.5, -0.5)
    assert numpy.allclose(surrogate_1.coefficients, surrogate_2.coefficients)
    assert numpy.isclose(surrogate_1(point), surrogate_2(point_shifted))


@pytest.mark.parametrize("linear_solver", ["lu", "lstsq", "inv"])
def test_surrogate_2(linear_solver):
    """Test if surrogate generates exact results for simple 1D function."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate((-10, 10), grid_gen)
    surrogate.train(function_2, linear_solver)
    # random point in the domain
    point = 0.367
    assert numpy.isclose(surrogate(point), function_2(point))


@pytest.mark.parametrize("linear_solver", ["lu", "lstsq", "inv"])
def test_surrogate_2_multi_input(linear_solver):
    """Test if surrogate generates exact results for 1D function and >1 input"""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate((-10, 10), grid_gen)
    surrogate.train(function_2, linear_solver)
    # random point in the domain
    numpy.random.seed(567)
    points_1 = numpy.array([0.367, 0.4, 0.742, 0.99])
    points_2 = numpy.random.rand(3, 4)
    assert numpy.allclose(surrogate(points_1), function_2(points_1))
    assert numpy.allclose(surrogate(points_2), function_2(points_2))


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_surrogate_1D_multi_shape(norm):
    """Test if surrogate call generates the correct shape for 1 dimension."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate((-10, 10), grid_gen, norm)
    surrogate.train(function_2)
    # arrays of various shapes with points in the domain
    assert numpy.isscalar(surrogate(1))
    assert surrogate([1]).shape == (1,)
    assert surrogate([[1]]).shape == (1,)
    assert surrogate([1, 2]).shape == (2,)
    assert surrogate([[1], [2]]).shape == (2,)
    assert surrogate(numpy.ones((1, 1, 1, 1))).shape == (1, 1, 1)
    assert surrogate(numpy.ones((1, 2, 3, 4))).shape == (1, 2, 3, 4)
    assert surrogate(numpy.ones((4, 3, 2, 1))).shape == (4, 3, 2)


def test_train_from_data():
    """Test if train_from_data_works."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4))
    surrogate = Surrogate([(-5, 10), (0, 15)], grid_gen)
    data = [branin(point) for point in surrogate.points]
    surrogate.train_from_data(data)
    # random point in the domain
    point = (8, 0.75)
    assert numpy.isclose(surrogate(point), branin(point))
    with pytest.raises(IndexError):
        surrogate.train_from_data(data[:5])


def test_gradient_train_from_data():
    """Test if train_from_data_works."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4))
    surrogate = GradientSurrogate([(-5, 10), (0, 15)], grid_gen)
    gradient_data = [function_3(point) for point in surrogate.points]
    surrogate.train_from_data(gradient_data)
    # random point in the domain
    point = (8, 0.75)
    assert numpy.allclose(surrogate.gradient(point), function_3(point))
    with pytest.raises(IndexError):
        surrogate.train_from_data(gradient_data[:5])


def test_gradient_surrogate_3():
    """Test if surrogate generates exact results for test for Chebyshevs."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([(-2, 2), (-2, 2)], grid_gen)
    surrogate.train(function_3)
    # random points in the domain
    points = [(0.649, -0.9), (-0.885, 1)]
    surrogate_gradient_values = [surrogate.gradient(x) for x in points]
    exact_values = [function_3(x) for x in points]
    assert numpy.allclose(surrogate_gradient_values, exact_values)


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_gradient_surrogate_3_multi_input(norm):
    """Test if surrogate generates exact results if gradient has >1 input."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([(-2, 2), (-2, 2)], grid_gen, norm)
    surrogate.train(function_3)
    # random points in the domain
    numpy.random.seed(567)
    points_1 = [(0.649, -0.9), (-0.885, 1)]
    points_2 = numpy.random.rand(3, 4, 2) * 4 - 2
    exact_values_1 = [function_3(x) for x in points_1]
    exact_values_2 = numpy.zeros(points_2.shape)
    for i in range(points_2.shape[0]):
        for j in range(points_2.shape[1]):
            exact_values_2[i, j] = function_3(tuple(points_2[i, j]))
    surrogate_gradient_values_1 = surrogate.gradient(points_1)
    surrogate_gradient_values_2 = surrogate.gradient(points_2)
    assert numpy.allclose(exact_values_1, surrogate_gradient_values_1)
    assert numpy.allclose(exact_values_2, surrogate_gradient_values_2)


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_gradient_surrogate_2D_multi_shape(norm):
    """Test if surrogate gradient generates correct shape for 2 dimensions."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([(-2, 2), (-2, 2)], grid_gen, norm)
    surrogate.train(function_3)
    # arrays of various shapes with points in the domain
    assert surrogate.gradient([1, 1]).shape == (2,)
    assert surrogate.gradient([[1, 1]]).shape == (1, 2)
    assert surrogate.gradient([[1, 1], [0.5, 0.5]]).shape == (2, 2)
    assert surrogate.gradient(numpy.ones((1, 1, 1, 2))).shape == (1, 1, 1, 2)
    assert surrogate.gradient(numpy.ones((4, 3, 2, 2))).shape == (4, 3, 2, 2)
    assert surrogate.gradient(numpy.ones((4, 3, 5, 2))).shape == (4, 3, 5, 2)


def test_gradient_surrogate_4():
    """Test if surrogate generates exact results for test for Chebyshevs."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([-2, 2], grid_gen)
    surrogate.train(function_4)
    # random points in the domain
    points = -0.7, 0.45
    surrogate_gradient_values = [surrogate.gradient(x) for x in points]
    exact_values = [function_4(x) for x in points]
    assert numpy.allclose(surrogate_gradient_values, exact_values)


def test_gradient_surrogate_4_multi_input():
    """Test if surrogate generates exact results for 1D with multiple inputs"""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([-2, 2], grid_gen)
    surrogate.train(function_4)
    # random points in the domain
    numpy.random.seed(567)
    points_1 = -0.7, 0.45
    points_2 = numpy.random.rand(3, 4)
    exact_values_1 = [function_4(x) for x in points_1]
    exact_values_2 = numpy.zeros(points_2.shape)
    for i in range(points_2.shape[0]):
        for j in range(points_2.shape[1]):
            exact_values_2[i, j] = function_4(points_2[i, j])
    surrogate_gradient_values_1 = surrogate.gradient(points_1)
    surrogate_gradient_values_2 = surrogate.gradient(points_2)
    assert numpy.allclose(surrogate_gradient_values_1, exact_values_1)
    assert numpy.allclose(surrogate_gradient_values_2, exact_values_2)


@pytest.mark.parametrize(
    "norm",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_gradient_surrogate_1D_multi_shape(norm):
    """Test if surrogate gradient generates correct shape for 1 dimension."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate((-2, 2), grid_gen, norm)
    surrogate.train(function_4)
    # arrays of various shapes with points in the domain
    assert numpy.isscalar(surrogate(1))
    assert surrogate([1]).shape == (1,)
    assert surrogate([[1]]).shape == (1,)
    assert surrogate([1, 2]).shape == (2,)
    assert surrogate([[1], [2]]).shape == (2,)
    assert surrogate(numpy.ones((1, 1, 1, 1))).shape == (1, 1, 1)
    assert surrogate(numpy.ones((1, 2, 3, 4))).shape == (1, 2, 3, 4)
    assert surrogate(numpy.ones((4, 3, 2, 1))).shape == (4, 3, 2)


def test_error_solver():
    """Test if invalid solver generates surrogate."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate((-10, 10), grid_gen)
    with pytest.raises(ValueError):
        surrogate.train(function_2, "inverse")


def test_error_function_needs_training():
    """Test if surrogate is generated without training."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen)
    # random point in the domain
    point = (0.649, 0)
    with pytest.raises(ValueError):
        surrogate(point)


def test_error_input_surrogate():
    """Test the error related to surrogate's input dimension."""
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen)
    surrogate.train(function_1, "lstsq")
    # random point in the domain but with incorrect dimensionality
    point = (0.649, 0, 0.5)
    point_outside_domain = (-2, 1)
    with pytest.raises(IndexError):
        surrogate(point)
    with pytest.raises(ValueError):
        surrogate(point_outside_domain)
    with pytest.raises(IndexError):
        surrogate.gradient(point)
    with pytest.raises(ValueError):
        surrogate.gradient(point_outside_domain)


@pytest.mark.parametrize(
    "change_norm",
    [
        SymmetricalLogNormalizer(),
        IntervalNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_norm_setter(change_norm):
    """Test if the changing the normalizer correctly retrains the function"""
    null_norm = NullNormalizer()
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = Surrogate([(-1, 1), (-1, 1)], grid_gen, null_norm)
    surrogate.train(function_1)
    null_coefficients = surrogate.coefficients
    # change norm
    surrogate.norm = change_norm
    change_coefficients = surrogate.coefficients
    # random points in the domain
    points = [(0.649, -0.9), (-0.885, 1)]
    surrogate_values = [surrogate(x) for x in points]
    exact_values = [function_1(x) for x in points]
    assert surrogate.norm == change_norm
    assert not numpy.array_equal(null_coefficients, change_coefficients)
    if isinstance(change_norm, SymmetricalLogNormalizer):
        assert numpy.allclose(surrogate_values, exact_values, atol=0.5, rtol=0.3)
    else:
        assert numpy.allclose(surrogate_values, exact_values)


@pytest.mark.parametrize(
    "change_norm",
    [
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_norm_setter_gradient(change_norm):
    """Test if the changing the normalizer correctly retrains the function"""
    null_norm = NullNormalizer()
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(2))
    surrogate = GradientSurrogate([(-1, 1), (-1, 1)], grid_gen, null_norm)
    surrogate.train(function_3)
    null_coefficients = surrogate.coefficients
    # change norm
    surrogate.norm = change_norm
    change_coefficients = surrogate.coefficients
    # random points in the domain
    points = [(0.649, -0.9), (-0.885, 1)]
    surrogate_values = surrogate.gradient(points)
    exact_values = [function_3(x) for x in points]
    assert surrogate.norm == change_norm
    assert not numpy.array_equal(null_coefficients, change_coefficients)
    if isinstance(change_norm, SymmetricalLogNormalizer):
        assert numpy.allclose(surrogate_values, exact_values, atol=0.5, rtol=0.3)
    else:
        assert numpy.allclose(surrogate_values, exact_values)
