import pytest

import numpy

import smolyay.samples

sample_points_answers = [
    (smolyay.samples.ClenshawCurtisPointSet(0), [0]),
    (smolyay.samples.ClenshawCurtisPointSet(1), [-1, 1]),
    (smolyay.samples.ClenshawCurtisPointSet(2), [-1, 0, 1]),
    (smolyay.samples.ClenshawCurtisPointSet(3), [-1, -0.5, 0.5, 1]),
    (
        smolyay.samples.ClenshawCurtisPointSet(8),
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
    (smolyay.samples.TrigonometricPointSet(0), [0]),
    (smolyay.samples.TrigonometricPointSet(1), [0]),
    (smolyay.samples.TrigonometricPointSet(2), [0, numpy.pi]),
    (smolyay.samples.TrigonometricPointSet(3), [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3]),
    (
        smolyay.samples.TrigonometricPointSet(9),
        2 * numpy.pi * numpy.linspace(0, 8 / 9, 9),
    ),
    (smolyay.samples.NestedClenshawCurtisPointSet(0), [0]),
    (smolyay.samples.NestedClenshawCurtisPointSet(1), [0, -1, 1]),
    (
        smolyay.samples.NestedClenshawCurtisPointSet(2),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.NestedClenshawCurtisPointSet(3),
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
    (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [0]),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [0, -1, 1]),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet(2),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet(3),
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
        smolyay.samples.SlowNestedClenshawCurtisPointSet(4),
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
        smolyay.samples.SlowNestedClenshawCurtisPointSet(5),
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
    (smolyay.samples.NestedTrigonometricPointSet(0), [0]),
    (
        smolyay.samples.NestedTrigonometricPointSet(1),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.NestedTrigonometricPointSet(2),
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
    "Non-nested Cheb [0]",
    "Non-nested Cheb [1]",
    "Non-nested Cheb [2]",
    "Non-nested Cheb [3]",
    "Non-nested Cheb [8]",
    "Non-nested Trig [0]",
    "Non-nested Trig [1]",
    "Non-nested Trig [2]",
    "Non-nested Trig [3]",
    "Non-nested Trig [9]",
    "Nested Cheb [0]",
    "Nested Cheb [1]",
    "Nested Cheb [2]",
    "Nested Cheb [3]",
    "Nested Slow Cheb [0]",
    "Nested Slow Cheb [1]",
    "Nested Slow Cheb [2]",
    "Nested Slow Cheb [3]",
    "Nested Slow Cheb [4]",
    "Nested Slow Cheb [5]",
    "Nested Trig [0]",
    "Nested Trig [1]",
    "Nested Trig [2]",
]

nested_sample_ids = [
    "Nested Cheb [0]",
    "Nested Cheb [1]",
    "Nested Cheb [2]",
    "Nested Cheb [3]",
    "Nested Slow Cheb [0]",
    "Nested Slow Cheb [1]",
    "Nested Slow Cheb [2]",
    "Nested Slow Cheb [3]",
    "Nested Slow Cheb [4]",
    "Nested Slow Cheb [5]",
    "Nested Trig [0]",
    "Nested Trig [1]",
    "Nested Trig [2]",
]


def test_clenshaw_initial():
    """test default properties"""
    f = smolyay.samples.ClenshawCurtisPointSet(7)
    assert numpy.array_equal(f.domain, [-1, 1])
    assert f.degree == 7


def test_trig_initial():
    """test default properties"""
    f = smolyay.samples.TrigonometricPointSet(7)
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])
    assert f.frequency == 7


@pytest.mark.parametrize(
    "nested_samples,domain",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet, [-1, 1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet, [-1, 1]),
        (smolyay.samples.NestedTrigonometricPointSet, [0, 2 * numpy.pi]),
    ],
    ids=["Nested Cheb", "Nested Slow Cheb", "Nested Trig"],
)
def test_nested_initial(nested_samples, domain):
    """test default properties"""
    f = nested_samples(7)
    assert numpy.array_equal(f.domain, domain)
    assert f.max_level == 7
    assert f._valid_cache == True
    f.points
    assert f._valid_cache == False


@pytest.mark.parametrize("object,points", sample_points_answers, ids=sample_points_ids)
def test_generate_points(object, points):
    """test the points of initialized UnidimensionalPointSet"""
    assert numpy.allclose(object.points, points, atol=1e-10)


@pytest.mark.parametrize(
    "nested_samples,num_points",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [1, 2]),
        (smolyay.samples.NestedClenshawCurtisPointSet(2), [1, 2, 2]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [1, 2]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(2), [1, 2, 2]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [1, 2, 2, 4, 0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [1, 2, 2, 4, 0, 8]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [1, 2]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [1, 2, 6]),
    ],
    ids=nested_sample_ids,
)
def test_nested_num_points_per_level(nested_samples, num_points):
    """test number of points per level"""
    assert nested_samples.num_points == num_points


@pytest.mark.parametrize(
    "nested_samples,start_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [0, 1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(2), [0, 1, 3]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [0, 1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(2), [0, 1, 3]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [0, 1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [0, 1, 3, 5, 9, 9]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [0]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [0, 1]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [0, 1, 3]),
    ],
    ids=nested_sample_ids,
)
def test_nested_start_level(nested_samples, start_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.start_level, start_level)


@pytest.mark.parametrize(
    "nested_samples,end_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [1, 3]),
        (smolyay.samples.NestedClenshawCurtisPointSet(2), [1, 3, 5]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [1, 3]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(2), [1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [1, 3, 5, 9, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [1, 3, 5, 9, 9, 17]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [1, 3]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [1, 3, 9]),
    ],
    ids=nested_sample_ids,
)
def test_nested_end_level(nested_samples, end_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.end_level, end_level)


@pytest.mark.parametrize(
    "nested_samples,level,points",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(3), 0, [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), 1, [-1, 1]),
        (
            smolyay.samples.NestedClenshawCurtisPointSet(3),
            2,
            [-1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet(3),
            3,
            [
                -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
                -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            ],
        ),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), 0, [0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), 1, [-1, 1]),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet(5),
            2,
            [-1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet(5),
            3,
            [
                -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
                -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
                numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            ],
        ),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), 4, []),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet(5),
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
        (smolyay.samples.NestedTrigonometricPointSet(2), 0, [0]),
        (
            smolyay.samples.NestedTrigonometricPointSet(2),
            1,
            [2 * numpy.pi / 3, 4 * numpy.pi / 3],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet(2),
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
