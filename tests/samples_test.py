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
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 1), [0]),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 2), [0, numpy.pi]),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 3),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 9),
        2 * numpy.pi * numpy.linspace(0, 8 / 9, 9),
    ),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 0), [0]),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [0, -1, 1]),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
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
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 0), [0]),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [0, -1, 1]),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3),
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
    (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 0), [0]),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
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
    "Trigonometric [2]",
    "Trigonometric [3]",
    "Trigonometric [9]",
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


def test_clenshaw_initial():
    """test default properties"""
    f = smolyay.samples.ClenshawCurtisPointSet([-2, 1], 7)
    assert numpy.array_equal(f.domain, [-2, 1])
    assert f.degree == 7


def test_trig_initial():
    """test default properties"""
    f = smolyay.samples.TrigonometricPointSet([0, 4 * numpy.pi], 7)
    assert numpy.array_equal(f.domain, [0, 4 * numpy.pi])
    assert f.frequency == 7


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=["NestedClenshawCurtis", "SlowNestedClenshawCurtis", "NestedTrigonometric"],
)
def test_nested_initial(nested_samples):
    """test default properties"""
    f = nested_samples([-10, 10], 7)
    assert numpy.array_equal(f.domain, [-10, 10])
    assert f.max_level == 7


@pytest.mark.parametrize(
    "samples",
    [
        smolyay.samples.ClenshawCurtisPointSet,
        smolyay.samples.TrigonometricPointSet,
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=[
        "ClenshawCurtis",
        "Trigonometric",
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_domain_error(samples):
    """test default properties"""
    with pytest.raises(TypeError):
        samples([-10, 10, 20], 7)
    with pytest.raises(TypeError):
        samples([[-10, 10], [-10, 10]], 7)


@pytest.mark.parametrize("samples,points", sample_points_answers, ids=sample_points_ids)
def test_generate_points(samples, points):
    """test the points of initialized UnidimensionalPointSet"""
    assert len(samples) == len(points)
    assert numpy.allclose(samples.points, points, atol=1e-10)
    new_points = numpy.array(points) * 5 + 10
    new_domain = samples.domain * 5 + 10
    samples.domain = new_domain
    assert numpy.array_equal(samples.domain, new_domain)
    assert numpy.allclose(samples.points, new_points, atol=1e-10)

@pytest.mark.parametrize(
    "nested_samples,num_per_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [1, 2]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2), [1, 2, 2]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [1, 2]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2), [1, 2, 2]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4), [1, 2, 2, 4, 0]),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            [1, 2, 2, 4, 0, 8],
        ),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1), [1, 2]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2), [1, 2, 6]),
    ],
    ids=nested_sample_ids,
)
def test_nested_num_per_level(nested_samples, num_per_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.num_per_level, num_per_level)


@pytest.mark.parametrize(
    "nested_samples,start_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 0), [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [0, 1]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2), [0, 1, 3]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 0), [0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [0, 1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2), [0, 1, 3]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4), [0, 1, 3, 5, 9]),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            [0, 1, 3, 5, 9, 9],
        ),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 0), [0]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1), [0, 1]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2), [0, 1, 3]),
    ],
    ids=nested_sample_ids,
)
def test_nested_start_level(nested_samples, start_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.start_level, start_level)


@pytest.mark.parametrize(
    "nested_samples,end_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [1, 3]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2), [1, 3, 5]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [1, 3]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2), [1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4), [1, 3, 5, 9, 9]),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            [1, 3, 5, 9, 9, 17],
        ),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1), [1, 3]),
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2), [1, 3, 9]),
    ],
    ids=nested_sample_ids,
)
def test_nested_end_level(nested_samples, end_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.end_level, end_level)


@pytest.mark.parametrize(
    "nested_samples,level,points",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3), 0, [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3), 1, [-1, 1]),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
            2,
            [-1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
            3,
            [
                -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
                -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            ],
        ),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5), 0, [0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5), 1, [-1, 1]),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            2,
            [-1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            3,
            [
                -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
                -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            ],
        ),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5), 4, []),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            5,
            [
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
        (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2), 0, [0]),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
            1,
            [2 * numpy.pi / 3, 4 * numpy.pi / 3],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
            2,
            2
            * numpy.pi
            * numpy.array(
                [
                    1 / 9,
                    2 / 9,
                    4 / 9,
                    5 / 9,
                    7 / 9,
                    8 / 9,
                ]
            ),
        ),
    ],
    ids=nested_sample_ids,
)
def test_nested_get_individual_levels(nested_samples, level, points):
    """test number of points per level"""
    end_lev = nested_samples.end_level[level]
    start_lev = nested_samples.start_level[level]
    assert numpy.allclose(nested_samples.points[start_lev:end_lev], points)
