import pytest

import numpy

import smolyay.samples
    

sample_points_answers = [
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 0), [0]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 1), [-1, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 2), [-1, 0, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 3), [-1, -0.5, 0.5, 1]),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 0), [0]),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 1), [0]),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 2), [0, numpy.pi]),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 3),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
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

all_sample_sets = [
        smolyay.samples.ClenshawCurtisPointSet,
        smolyay.samples.TrigonometricPointSet,
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
]

sample_points_ids = [
    "Non-nested Cheb [0]",
    "Non-nested Cheb [1]",
    "Non-nested Cheb [2]",
    "Non-nested Cheb [3]",
    "Non-nested Trig [0]",
    "Non-nested Trig [1]",
    "Non-nested Trig [2]",
    "Non-nested Trig [3]",
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
    f = smolyay.samples.ClenshawCurtisPointSet([-1, 1], 7)
    assert numpy.array_equal(f.domain, [-1, 1])
    assert f.degree == 7
    assert f._valid_cache == False


def test_trig_initial():
    """test default properties"""
    f = smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 7)
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])
    assert f.frequency == 7
    assert f._valid_cache == False


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=["Nested Cheb", "Nested Slow Cheb", "Nested Trig"],
)
def test_nested_initial(nested_samples):
    """test default properties"""
    f = nested_samples([-10, 10],7)
    assert numpy.array_equal(f.domain, [-10, 10])
    assert f.max_level == 7
    assert f._valid_cache == False

@pytest.mark.parametrize(
    "samples",
    all_sample_sets,
    ids=[
        "Unnested Cheb",
        "Unnested Trig",
        "Nested Cheb",
        "Nested Slow Cheb",
        "Nested Trig"],
)
def test_set_domain(samples):
    """test default properties"""
    f = samples([-10, 10],7)
    f.points
    assert f._valid_cache == True
    f.domain = [-100, 100]
    assert f._valid_cache == False
    assert numpy.array_equal(f.domain, [-100, 100])

@pytest.mark.parametrize(
    "samples",
    all_sample_sets,
    ids=[
        "Unnested Cheb",
        "Unnested Trig",
        "Nested Cheb",
        "Nested Slow Cheb",
        "Nested Trig"],
)
def test_domain_error(samples):
    """test default properties"""
    with pytest.raises(TypeError):
        samples([-10, 5, 10],7)
    with pytest.raises(ValueError):
        samples([10, -10],7)


@pytest.mark.parametrize("samples,points", sample_points_answers, ids=sample_points_ids)
def test_generate_points(samples, points):
    """test the points of initialized UnidimensionalPointSet"""
    assert samples._valid_cache == False
    assert numpy.allclose(samples.points, points, atol=1e-10)
    assert samples._valid_cache == True


@pytest.mark.parametrize("samples,points", sample_points_answers, ids=sample_points_ids)
def test_len(samples, points):
    """test the len method of UnidimensionalPointSet objects"""
    assert len(samples) == len(points)

@pytest.mark.parametrize(
    "nested_samples,num_points_per_level",
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
def test_nested_num_points_per_level(nested_samples, num_points_per_level):
    """test number of points per level"""
    assert nested_samples._valid_cache == False
    assert numpy.array_equal(nested_samples.num_points_per_level, num_points_per_level)
    assert nested_samples._valid_cache == True


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
    assert nested_samples._valid_cache == False
    assert numpy.array_equal(nested_samples.start_level, start_level)
    assert nested_samples._valid_cache == True


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
    assert nested_samples._valid_cache == False
    assert numpy.array_equal(nested_samples.end_level, end_level)
    assert nested_samples._valid_cache == True
