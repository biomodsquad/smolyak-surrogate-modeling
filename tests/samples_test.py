import numpy
import pytest

import smolyay

sample_points_answers = [
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 0), [0]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 1), [-1, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 2), [-1, 0, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 3), [-1, -0.5, 0.5, 1]),
    (
        smolyay.samples.ClenshawCurtisPointSet([-1, 1], 8),
        [
            -1,
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            0,
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            1 / numpy.sqrt(2),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            1,
        ],
    ),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 0), [0]),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 1),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 4),
        2 * numpy.pi * numpy.linspace(0, 8 / 9, 9),
    ),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [0]),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2), [0, -1, 1]),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 4),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [0]),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2), [0, -1, 1]),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 6),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(2 + numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 + numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 - numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 - numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            numpy.sqrt(2 - numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            numpy.sqrt(2 - numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            numpy.sqrt(2 + numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            numpy.sqrt(2 + numpy.sqrt(2 + numpy.sqrt(2))) / 2,
        ],
    ),
    (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1), [0]),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 3),
        2
        * numpy.pi
        * numpy.array(
            [
                0,
                1 / 3,
                2 / 3,
                1 / 9,
                2 / 9,
                4 / 9,
                5 / 9,
                7 / 9,
                8 / 9,
            ]
        ),
    ),
]

sample_points_ids = [
    "ClenshawCurtis [0]",
    "ClenshawCurtis [1]",
    "ClenshawCurtis [2]",
    "ClenshawCurtis [3]",
    "ClenshawCurtis [8]",
    "Trigonometric [0]",
    "Trigonometric [1]",
    "Trigonometric [4]",
    "NestedClenshawCurtis [0]",
    "NestedClenshawCurtis [1]",
    "NestedClenshawCurtis [2]",
    "NestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [0]",
    "SlowNestedClenshawCurtis [1]",
    "SlowNestedClenshawCurtis [2]",
    "SlowNestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [4]",
    "SlowNestedClenshawCurtis [5]",
    "NestedTrigonometric [0]",
    "NestedTrigonometric [1]",
    "NestedTrigonometric [2]",
]

nested_sample_ids = [
    "NestedClenshawCurtis [0]",
    "NestedClenshawCurtis [1]",
    "NestedClenshawCurtis [2]",
    "NestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [0]",
    "SlowNestedClenshawCurtis [1]",
    "SlowNestedClenshawCurtis [2]",
    "SlowNestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [4]",
    "SlowNestedClenshawCurtis [5]",
    "NestedTrigonometric [0]",
    "NestedTrigonometric [1]",
    "NestedTrigonometric [2]",
]


def test_initialize_clenshaw():
    """test default properties"""
    f = smolyay.samples.ClenshawCurtisPointSet([-2, 1], 3)
    assert numpy.array_equal(f.domain, [-2, 1])
    assert f.degree == 3
    assert isinstance(f.degree, int)
    f.degree = float(5)
    assert f.degree == 5
    assert isinstance(f.degree, int)


def test_degree_error():
    """test degree error given invalid degree"""
    with pytest.raises(ValueError):
        smolyay.samples.ClenshawCurtisPointSet([-2, 1], -7)
    with pytest.raises(ValueError):
        f = smolyay.samples.ClenshawCurtisPointSet([-2, 1], 3)
        f.degree = -5


def test_initialize_trig():
    """test default properties"""
    f = smolyay.samples.TrigonometricPointSet([0, 4 * numpy.pi], 3)
    assert numpy.array_equal(f.domain, [0, 4 * numpy.pi])
    assert f.frequency == 3
    assert isinstance(f.frequency, int)
    f.frequency = float(-5)
    assert f.frequency == 5
    assert isinstance(f.frequency, int)


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_initialize_nested(nested_samples):
    """test default properties"""
    f = nested_samples([-10, 10], 4)
    assert numpy.array_equal(f.domain, [-10, 10])
    assert f.num_levels == 4
    assert isinstance(f.num_levels, int)
    f.num_levels = float(5)
    assert f.num_levels == 5
    assert isinstance(f.num_levels, int)


@pytest.mark.parametrize(
    "samples,set_args",
    [
        (smolyay.samples.ClenshawCurtisPointSet, {"degree": 4}),
        (smolyay.samples.TrigonometricPointSet, {"frequency": 4}),
        (smolyay.samples.NestedClenshawCurtisPointSet, {"num_levels": 4}),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet, {"num_levels": 4}),
        (smolyay.samples.NestedTrigonometricPointSet, {"num_levels": 4}),
    ],
    ids=[
        "ClenshawCurtis",
        "Trigonometric",
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_domain_error(samples, set_args):
    """test error given invalid domain and that reversed domains swap"""
    # reverse domain
    f = samples([10, -10], **set_args)
    assert numpy.array_equal(f.domain, [-10, 10])
    f = samples([-10, 10], **set_args)
    f.domain = [5, -10]
    assert numpy.array_equal(f.domain, [-10, 5])
    # invalid domain
    with pytest.raises(TypeError):
        samples([-10, 10, 20], **set_args)
    with pytest.raises(TypeError):
        samples([[-10, 10], [-10, 10]], **set_args)
    with pytest.raises(ValueError):
        samples([10, 10], **set_args)
    with pytest.raises(TypeError):
        f = samples([-10, 10], **set_args)
        f.domain = [[-10, 10], [-10, 10]]
    with pytest.raises(TypeError):
        f = samples([-10, 10], **set_args)
        f.domain = [-10, 10, 20]
    with pytest.raises(ValueError):
        f = samples([-10, 10], **set_args)
        f.domain = [10, 10]


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_num_levels_error(nested_samples):
    """test error given invalid num_levels"""
    with pytest.raises(ValueError):
        f = nested_samples([-10, 10], 0)
    f = nested_samples([-10, 10], 2)
    with pytest.raises(ValueError):
        f.num_levels = 0


@pytest.mark.parametrize("samples,points", sample_points_answers, ids=sample_points_ids)
def test_generate_points(samples, points):
    """test the points of initialized UnidimensionalPointSet"""
    assert len(samples) == len(points)
    assert numpy.allclose(samples.points, points, atol=1e-10)

    # test for different domain and that points update after changing domain
    new_points = numpy.array(points) * 5 + 10
    new_domain = samples.domain * 5 + 10
    samples.domain = new_domain
    assert numpy.array_equal(samples.domain, new_domain)
    assert numpy.allclose(samples.points, new_points, atol=1e-10)
    assert samples[0] == pytest.approx(new_points[0])
    assert numpy.allclose(samples[:], new_points, atol=1e-10)
    assert numpy.allclose(list(samples), new_points, atol=1e-10)


@pytest.mark.parametrize(
    "nested_samples,num_per_level,start_level,end_level",
    [
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            [1, 2, 2, 4, 0],
            [0, 1, 3, 5, 9],
            [1, 3, 5, 9, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 6),
            [1, 2, 2, 4, 0, 8],
            [0, 1, 3, 5, 9, 9],
            [1, 3, 5, 9, 9, 17],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 3),
            [1, 2, 6],
            [0, 1, 3],
            [1, 3, 9],
        ),
    ],
    ids=nested_sample_ids,
)
def test_nested_levels(nested_samples, num_per_level, start_level, end_level):
    """test number of points per level, start level indexes, and end level indexes"""
    assert numpy.array_equal(nested_samples.num_per_level, num_per_level)
    assert numpy.array_equal(nested_samples.start_level, start_level)
    assert numpy.array_equal(nested_samples.end_level, end_level)
    for level, (start, end) in enumerate(
        zip(nested_samples.start_level, nested_samples.end_level)
    ):
        assert numpy.allclose(nested_samples.level(level), nested_samples[start:end])
