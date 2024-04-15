import pytest

import numpy

from smolyay.samples import (
    UnidimensionalPointSet,
    ClenshawCurtisPointSet,NestedClenshawCurtisPointSet,
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
    assert f.number_points == 0


def test_clenshaw_num3():
    """test number of points is changed to 3"""
    f = ClenshawCurtisPointSet()
    f.number_points = 3
    assert numpy.allclose(f.points, [-1, 0, 1], atol=1e-10)
    assert f.number_points == 3


def test_clenshaw_num_increased(clenshaw_curtis_num9):
    """test number of points is increased"""
    f = ClenshawCurtisPointSet()
    f.number_points = 3
    f.number_points = 4
    assert numpy.allclose(f.points, [-1, -0.5, 0.5, 1], atol=1e-10)
    assert f.number_points == 4


def test_clenshaw_num_decreased(clenshaw_curtis_num9):
    """test number of points is decreased"""
    f = ClenshawCurtisPointSet()
    f.number_points = 5
    f.number_points = 2
    assert numpy.allclose(f.points, [-1, 1], atol=1e-10)
    assert f.number_points == 2


def test_clenshaw_num_error():
    """test number of points is invalid"""
    f = ClenshawCurtisPointSet()
    with pytest.raises(TypeError):
        f.number_points = "Not a number"
    with pytest.raises(ValueError):
        f.number_points = -1

def test_nestedclenshaw_initial():
    """test default properties"""
    f = ClenshawCurtisPointSet()
    assert f.domain == [-1, 1]
    assert f.points == []
    assert f.number_points == 0


def test_nestedclenshaw_num3():
    """test number of points is changed to 3"""
    f = NestedClenshawCurtisPointSet()
    f.number_points = 3
    assert numpy.allclose(f.points, [0, -1, 1], atol=1e-10)
    assert f.number_points == 3


def test_nestedclenshaw_num_increased(clenshaw_curtis_num9):
    """test number of points is increased"""
    f = NestedClenshawCurtisPointSet()
    f.number_points = 3
    f.number_points = 9
    assert numpy.allclose(f.points, clenshaw_curtis_num9, atol=1e-10)
    assert f.number_points == 9


def test_nestedclenshaw_num_decreased(clenshaw_curtis_num9):
    """test number of points is decreased"""
    f = NestedClenshawCurtisPointSet()
    f.number_points = 16
    f.number_points = 9
    assert numpy.allclose(f.points, clenshaw_curtis_num9, atol=1e-10)
    assert f.number_points == 9


def test_nestedclenshaw_num_error():
    """test number of points is invalid"""
    f = NestedClenshawCurtisPointSet()
    with pytest.raises(TypeError):
        f.number_points = "Not a number"
    with pytest.raises(ValueError):
        f.number_points = -1


def test_trig_initial():
    """test default properties"""
    f = TrigonometricPointSet()
    assert f.domain == [0, 2 * numpy.pi]
    assert f.points == []
    assert f.number_points == 0


def test_trig_num3():
    """test number of points is changed to 3"""
    f = TrigonometricPointSet()
    f.number_points = 3
    assert numpy.allclose(
        f.points, 2 * numpy.pi * numpy.array([0, 1 / 3, 2 / 3]), atol=1e-10
    )
    assert f.number_points == 3


def test_trig_num_increased(trig_num9):
    """test number of points is increased"""
    f = TrigonometricPointSet()
    f.number_points = 3
    f.number_points = 9
    assert numpy.allclose(f.points, trig_num9, atol=1e-10)
    assert f.number_points == 9


def test_trig_num_decreased(trig_num9):
    """test number of points is decreased"""
    f = TrigonometricPointSet()
    f.number_points = 16
    f.number_points = 9
    assert numpy.allclose(f.points, trig_num9, atol=1e-10)
    assert f.number_points == 9


def test_trig_num_error():
    """test number of points is invalid"""
    f = TrigonometricPointSet()
    with pytest.raises(TypeError):
        f.number_points = "Not a number"
    with pytest.raises(ValueError):
        f.number_points = -1
