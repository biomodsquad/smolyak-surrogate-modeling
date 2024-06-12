import numpy
import pytest
import scipy.stats.qmc

import smolyay



def test_map_function_1d():
    """Test if points are transfromed properly."""
    domain = (-8, 12)
    new_domain = (0, 1)
    assert smolyay.grid.MultidimensionalPointSet._scale_to_domain(2, domain, new_domain)
    assert numpy.allclose(
        smolyay.grid.MultidimensionalPointSet._scale_to_domain([-3, 7], domain, new_domain),
        [0.25, 0.75],
    )


def test_map_function_2d():
    """Test if points are transfromed properly."""
    domain = ((-10, 10), (0, 2))
    new_domain = ((0, 1), (-1, 1))
    assert numpy.allclose(
        smolyay.grid.MultidimensionalPointSet._scale_to_domain((-10, 0), domain, new_domain), [0, -1]
    )
    assert numpy.allclose(
        smolyay.grid.MultidimensionalPointSet._scale_to_domain(
            [(0, 1), (10, 2)], domain, new_domain
        ),
        [[0.5, 0], [1, 1]],
    )


def test_random_initalize_without_optional():
    """Test that the random point set initializes correctly"""
    f = smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 70, "latin", 1234)
    assert numpy.array_equal(f.domain, [[-10, 10], [0, 2]])
    assert f.dimension == 2
    assert f.number_points == 70
    assert f.method == "latin"
    assert f.seed == 1234
    assert f.options == None


def test_random_initalize_with_optional():
    """Test that the random point set initializes correctly"""
    f2 = smolyay.grid.RandomPointSet([[-10, 20]], 49, "latin", 5678, options={"strength": 2})
    assert numpy.array_equal(f2.domain, [[-10, 20]])
    assert f2.dimension == 1
    assert f2.number_points == 49
    assert f2.method == "latin"
    assert f2.seed == 5678
    assert f2.options == {"strength": 2}


def test_random_domain_error():
    """Test that a IndexError is given if the domain is invalid"""
    with pytest.raises(IndexError):
        smolyay.grid.RandomPointSet([[-10, 10, 11], [0, 2, 11]], 70, "latin", 1234)


def test_random_number_points_error():
    """Test that a TypeError is given if the number of points is invalid"""
    with pytest.raises(TypeError):
        smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 70.6, "latin", 5)


@pytest.mark.parametrize(
    "method",
    [
        "latin",
        "halton",
        "sobol",
        "uniform",
        "LATIN",
        "HALTON",
        "SOBOL",
        "UNIFORM",
        "Latin",
        "Halton",
        "Sobol",
        "Uniform",
    ],
)
def test_random_method_case_insensitive(method):
    """Test that a TypeError is given if the seed is invalid"""
    f = smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 70, method, 4)
    assert f.method == method.lower()


def test_random_method_error():
    """Test that a TypeError is given if the seed is invalid"""
    with pytest.raises(ValueError):
        smolyay.grid.RandomPointSet([[-10, 10]], 70, "not a method", 4.5)


def test_random_seed_error():
    """Test that a TypeError is given if the seed is invalid"""
    with pytest.raises(TypeError):
        smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 70, "latin", 4.5)


def test_random_points():
    """Test that the random point set generates the correct random points"""
    p_gen = scipy.stats.qmc.LatinHypercube(2, seed=1234).random(n=70)
    new_points = scipy.stats.qmc.scale(p_gen, [-10, 0], [10, 2])
    f = smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 70, "latin", 1234)
    assert numpy.array_equal(f.points, new_points)


def test_random_options():
    """Test that the options parameter is passed to the QMCEngine"""
    p_gen = scipy.stats.qmc.LatinHypercube(2, seed=1234, strength=2).random(n=49)
    new_points = scipy.stats.qmc.scale(p_gen, [-10, 0], [10, 2])
    f = smolyay.grid.RandomPointSet([[-10, 10], [0, 2]], 49, "latin", 1234, options={"strength": 2})
    assert numpy.array_equal(f.points, new_points)


def test_random_sobol_error():
    """Test classmethod error using sobol if number of points not a power of 2"""
    p_gen = scipy.stats.qmc.Sobol(1, seed=1234).random(n=16)
    ref_points = scipy.stats.qmc.scale(p_gen, [0], [2])
    test_points = smolyay.grid.RandomPointSet.get_random_points([[0, 2]], 16, "sobol", 1234)
    assert numpy.array_equal(ref_points, test_points)
    with pytest.raises(ValueError):
        smolyay.grid.RandomPointSet.get_random_points([[0, 2]], 70, "sobol", 1234)


def test_random_set_domain():
    """Test that set_params sets the parameters"""
    p_gen = scipy.stats.qmc.Halton(3, seed=4).random(n=5)
    points = scipy.stats.qmc.scale(p_gen, [-10, 0, 0], [10, 2, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    f.set_params(domain=[[-10, 10], [0, 2], [0, 9]])
    assert numpy.array_equal(f.domain, [[-10, 10], [0, 2], [0, 9]])
    assert f.dimension == 3
    assert numpy.array_equal(f.points, points)


def test_random_set_number_points():
    """Test that set_params sets the parameters"""
    p_gen = scipy.stats.qmc.Halton(2, seed=4).random(n=70)
    points = scipy.stats.qmc.scale(p_gen, [-3, 6], [5, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    f.set_params(number_points=70)
    assert f.number_points == 70
    assert numpy.array_equal(f.points, points)


def test_random_set_method():
    """Test that set_params sets the parameters"""
    p_gen = scipy.stats.qmc.LatinHypercube(2, seed=4).random(n=5)
    points = scipy.stats.qmc.scale(p_gen, [-3, 6], [5, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    f.set_params(method="latin")
    assert f.method == "latin"
    assert numpy.array_equal(f.points, points)


def test_random_set_seed():
    """Test that set_params sets the parameters"""
    p_gen = scipy.stats.qmc.Halton(2, seed=1234).random(n=5)
    points = scipy.stats.qmc.scale(p_gen, [-3, 6], [5, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    f.set_params(seed=1234)
    assert f.seed == 1234
    assert numpy.array_equal(f.points, points)


def test_random_set_options():
    """Test that set_params sets the parameters"""
    p_gen = scipy.stats.qmc.Halton(2, seed=4, scramble=False).random(n=5)
    points = scipy.stats.qmc.scale(p_gen, [-3, 6], [5, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    f.set_params(options={"scramble": False})
    assert f.options == {"scramble": False}
    assert numpy.array_equal(f.points, points)


def test_random_set_params():
    """Test that set_params sets the parameters"""
    p_gen1 = scipy.stats.qmc.Halton(2, seed=4).random(n=5)
    new_points1 = scipy.stats.qmc.scale(p_gen1, [-3, 6], [5, 9])
    p_gen2 = scipy.stats.qmc.LatinHypercube(3, seed=1234).random(n=70)
    new_points2 = scipy.stats.qmc.scale(p_gen2, [-10, 0, 0], [10, 2, 9])
    f = smolyay.grid.RandomPointSet([[-3, 5], [6, 9]], 5, "halton", 4)
    assert numpy.array_equal(f.points, new_points1)
    f.set_params(
        domain=[[-10, 10], [0, 2], [0, 9]], method="latin", seed=1234, number_points=70
    )
    assert numpy.array_equal(f.domain, [[-10, 10], [0, 2], [0, 9]])
    assert f.dimension == 3
    assert f.seed == 1234
    assert f.number_points == 70
    assert f.method == "latin"
    assert f.options == None
    assert numpy.array_equal(f.points, new_points2)


@pytest.mark.parametrize(
    "domain,num_points,method,seed,answer",
    [
        (
            [[-3, 5], [6, 9]],
            5,
            "halton",
            4,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Halton(2, seed=4).random(n=5), [-3, 6], [5, 9]
            ),
        ),
        (
            [[-10, 10], [0, 2], [0, 9]],
            70,
            "latin",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.LatinHypercube(3, seed=1234).random(n=70),
                [-10, 0, 0],
                [10, 2, 9],
            ),
        ),
        (
            [[-10, 10], [0, 9]],
            32,
            "sobol",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Sobol(2, seed=1234).random(n=32), [-10, 0], [10, 9]
            ),
        ),
        (
            [[-10, 10], [0, 9], [0, 1], [0, 1]],
            100,
            "uniform",
            1234,
            scipy.stats.qmc.scale(
                numpy.random.default_rng(seed=1234).uniform(size=(100, 4)),
                [-10, 0, 0, 0],
                [10, 9, 1, 1],
            ),
        ),
    ],
    ids=["Halton", "LatinHypercube", "Sobol", "Uniform"],
)
def test_random_classmethod(domain, num_points, method, seed, answer):
    """Test each method for generating random points"""
    points = smolyay.grid.RandomPointSet.get_random_points(domain, num_points, method, seed)
    assert numpy.array_equal(points, answer)


@pytest.mark.parametrize(
    "domain,num_points,method,seed,answer",
    [
        (
            [[-3, 5], [6, 9]],
            5,
            "HALTON",
            4,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Halton(2, seed=4).random(n=5), [-3, 6], [5, 9]
            ),
        ),
        (
            [[-10, 10], [0, 2], [0, 9]],
            70,
            "LATIN",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.LatinHypercube(3, seed=1234).random(n=70),
                [-10, 0, 0],
                [10, 2, 9],
            ),
        ),
        (
            [[-10, 10], [0, 9]],
            32,
            "SOBOL",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Sobol(2, seed=1234).random(n=32), [-10, 0], [10, 9]
            ),
        ),
        (
            [[-10, 10], [0, 9], [0, 1], [0, 1]],
            100,
            "UNIFORM",
            1234,
            scipy.stats.qmc.scale(
                numpy.random.default_rng(seed=1234).uniform(size=(100, 4)),
                [-10, 0, 0, 0],
                [10, 9, 1, 1],
            ),
        ),
        (
            [[-3, 5], [6, 9]],
            5,
            "Halton",
            4,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Halton(2, seed=4).random(n=5), [-3, 6], [5, 9]
            ),
        ),
        (
            [[-10, 10], [0, 2], [0, 9]],
            70,
            "Latin",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.LatinHypercube(3, seed=1234).random(n=70),
                [-10, 0, 0],
                [10, 2, 9],
            ),
        ),
        (
            [[-10, 10], [0, 9]],
            32,
            "Sobol",
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Sobol(2, seed=1234).random(n=32), [-10, 0], [10, 9]
            ),
        ),
        (
            [[-10, 10], [0, 9], [0, 1], [0, 1]],
            100,
            "Uniform",
            1234,
            scipy.stats.qmc.scale(
                numpy.random.default_rng(seed=1234).uniform(size=(100, 4)),
                [-10, 0, 0, 0],
                [10, 9, 1, 1],
            ),
        ),
    ],
)
def test_random_classmethod_case_insensitive(domain, num_points, method, seed, answer):
    """Test each method for generating random points"""
    points = smolyay.grid.RandomPointSet.get_random_points(domain, num_points, method, seed)
    assert numpy.array_equal(points, answer)


def test_random_classmethod_invalid_method():
    """Test classmethod get_random_points returns ValueError for invalid method"""
    with pytest.raises(ValueError):
        smolyay.grid.RandomPointSet.get_random_points([[0, 6], [1, 5]], 100, "method", 1234)


def test_generate_tensor_points():
    """Test the generate_tensor_combinations for a series with multiple sets."""
    point_sets = [numpy.array([9, 8, 7]), numpy.array([1, 2])]
    answer = [[9, 1], [9, 2], [8, 1], [8, 2], [7, 1], [7, 2]]
    f = smolyay.grid.TensorPointSet(point_sets)
    assert numpy.array_equal(f.points, answer)


def test_generate_smolyak_points():
    """Test the generate_tensor_combinations for a series with one set."""
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 3),
    ]
    answer = [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -2.0],
        [0.0, 2.0],
        [-0.70710678, 0.0],
        [0.70710678, 0.0],
        [-1.0, -2.0],
        [-1.0, 2.0],
        [1.0, -2.0],
        [1.0, 2.0],
        [0.0, -1.41421356],
        [0.0, 1.41421356],
    ]
    f = smolyay.grid.SmolyakPointSet(point_sets)
    assert numpy.allclose(f.points, answer)


def test_generate_compositions_include_zero_true():
    """Test the generate compositions function if include_zero is true."""
    composition_expected = [[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]
    composition_obtained = []
    composition_obtained = list(smolyay.grid.generate_compositions(6, 2, include_zero=True))
    assert composition_obtained == composition_expected


def test_generate_compositions_include_zero_false():
    """Test the generate compositions function if include_zero is false."""
    composition_expected = [[5, 1], [4, 2], [3, 3], [2, 4], [1, 5]]
    composition_obtained = list(smolyay.grid.generate_compositions(6, 2, include_zero=False))
    assert composition_obtained == composition_expected


def test_generate_compositions_zero_false_error():
    """Test that generate compositions raises an error for invalid input."""
    with pytest.raises(ValueError):
        list(smolyay.grid.generate_compositions(6, 7, include_zero=False))
