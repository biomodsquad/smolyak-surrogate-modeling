import pytest

import numpy

import smolyay.samples


def test_clenshaw_initial():
    """test default properties"""
    f = smolyay.samples.ClenshawCurtisPointSet(7)
    assert numpy.array_equal(f.domain, [-1, 1])
    assert f.degree == 7


@pytest.mark.parametrize(
    "degree,points",
    [
        (0, [0]),
        (1, [-1, 1]),
        (2, [-1, 0, 1]),
        (3, [-1, -0.5, 0.5, 1]),
        (
            8,
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
    ],
)
def test_clenshaw_generate_values(degree, points):
    """test the output of generate_points"""
    assert numpy.allclose(
        smolyay.samples.ClenshawCurtisPointSet(degree).points, points, atol=1e-10
    )


def test_trig_initial():
    """test default properties"""
    f = smolyay.samples.TrigonometricPointSet(7)
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])
    assert f.frequency == 7


@pytest.mark.parametrize(
    "degree,points",
    [
        (0, [0]),
        (1, [0]),
        (2, [0, numpy.pi]),
        (3, [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3]),
        (
            9,
            2 * numpy.pi * numpy.linspace(0, 8 / 9, 9),
        ),
    ],
)
def test_trig_generate_values(degree, points):
    """test number of points is changed to 3"""
    assert numpy.allclose(
        smolyay.samples.TrigonometricPointSet(degree).points, points, atol=1e-10
    )


@pytest.mark.parametrize(
    "nested_samples,domain",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet, [-1, 1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet, [-1, 1]),
        (smolyay.samples.NestedTrigonometricPointSet, [0, 2 * numpy.pi]),
    ],
)
def test_nested_initial(nested_samples, domain):
    """test default properties"""
    f = nested_samples(7)
    assert numpy.array_equal(f.domain, domain)
    assert f.max_level == 7
    assert f._valid_cache == True
    f.points
    assert f._valid_cache == False


@pytest.mark.parametrize(
    "nested_samples,points",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [0, -1, 1]),
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
    ],
)
def test_nested_points(nested_samples, points):
    """test number of points"""
    assert numpy.allclose(nested_samples.points, points, atol=1e-10)


@pytest.mark.parametrize(
    "nested_samples,num_points",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [1, 2]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [1, 2]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [1, 2, 2, 4]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [1, 2, 2, 4, 0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [1, 2, 2, 4, 0, 8]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [1, 2]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [1, 2, 6]),
    ],
)
def test_nested_num_points_per_level(nested_samples, num_points):
    """test number of points per level"""
    assert nested_samples.num_points == num_points

@pytest.mark.parametrize(
    "nested_samples,start_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [0]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [0, 1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [0]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [0, 1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [0, 1, 3, 5]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [0, 1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [0, 1, 3, 5, 9, 9]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [0]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [0, 1]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [0, 1, 3]),
    ],
)
def test_nested_start_level(nested_samples, start_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.start_level,start_level)

@pytest.mark.parametrize(
    "nested_samples,end_level",
    [
        (smolyay.samples.NestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.NestedClenshawCurtisPointSet(1), [1, 3]),
        (smolyay.samples.NestedClenshawCurtisPointSet(3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(0), [1]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(1), [1, 3]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(3), [1, 3, 5, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(4), [1, 3, 5, 9, 9]),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet(5), [1, 3, 5, 9, 9, 17]),
        (smolyay.samples.NestedTrigonometricPointSet(0), [1]),
        (smolyay.samples.NestedTrigonometricPointSet(1), [1, 3]),
        (smolyay.samples.NestedTrigonometricPointSet(2), [1, 3, 9]),
    ],
)
def test_nested_end_level(nested_samples, end_level):
    """test number of points per level"""
    assert numpy.array_equal(nested_samples.end_level,end_level)