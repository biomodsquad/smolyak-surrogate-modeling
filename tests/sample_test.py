import pytest

import numpy

from smolyay.samples import (
    UnidimensionalPointSet,
    ClenshawCurtisPointSet,
    NestedClenshawCurtisPointSet,
    TrigonometricPointSet,
)


@pytest.fixture
def clenshaw_curtis_num9():
    """the first 9 extrema"""
    return [
        0,
        -1.0,
        1.0,
        -1 / numpy.sqrt(2),
        1 / numpy.sqrt(2),
        -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
    ]


@pytest.fixture
def trig_num9():
    """the first 9 points"""
    return (
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
        )
    )


def test_clenshaw_initial():
    """test default properties"""
    f = ClenshawCurtisPointSet()
    assert f.domain == [-1, 1]
    assert f.points == []


def test_clenshaw_generate_values():
    """test the output of generate_points"""
    f = ClenshawCurtisPointSet()
    assert numpy.allclose(f.generate_points(0).points, [], atol=1e-10)
    assert numpy.allclose(f.generate_points(1).points, [0], atol=1e-10)
    assert numpy.allclose(f.generate_points(2).points, [-1, 1], atol=1e-10)
    assert numpy.allclose(f.generate_points(3).points, [-1, 0, 1], atol=1e-10)
    assert numpy.allclose(f.generate_points(4).points, [-1, -0.5, 0.5, 1], atol=1e-10)


def test_nestedclenshaw_initial():
    """test default properties"""
    f = ClenshawCurtisPointSet()
    assert f.domain == [-1, 1]
    assert f.points == []


def test_nestedclenshaw_num3(clenshaw_curtis_num9):
    """test number of points is changed to 3"""
    f = NestedClenshawCurtisPointSet()
    assert numpy.allclose(f.generate_points(0).points, [], atol=1e-10)
    assert numpy.allclose(f.generate_points(1).points, [0], atol=1e-10)
    assert numpy.allclose(f.generate_points(2).points, [0, -1], atol=1e-10)
    assert numpy.allclose(f.generate_points(3).points, [0, -1, 1], atol=1e-10)
    assert numpy.allclose(
        f.generate_points(4).points, [0, -1, 1, -1 / numpy.sqrt(2)], atol=1e-10
    )
    assert numpy.allclose(
        f.generate_points(9).points, [clenshaw_curtis_num9], atol=1e-10
    )


def test_trig_initial():
    """test default properties"""
    f = TrigonometricPointSet()
    assert f.domain == [0, 2 * numpy.pi]
    assert f.points == []


def test_trig_generate_values(trig_num9):
    """test number of points is changed to 3"""
    f = TrigonometricPointSet()
    assert numpy.allclose(f.generate_points(0).points, [], atol=1e-10)
    assert numpy.allclose(f.generate_points(1).points, [0], atol=1e-10)
    assert numpy.allclose(
        f.generate_points(2).points, [0, 2 * numpy.pi / 3], atol=1e-10
    )
    assert numpy.allclose(
        f.generate_points(3).points, [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3], atol=1e-10
    )
    assert numpy.allclose(f.generate_points(9).points, trig_num9, atol=1e-10)
