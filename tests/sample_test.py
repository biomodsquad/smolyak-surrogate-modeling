import pytest

import numpy

from smolyay.samples import (
    UnidimensionalPointSet,
    ClenshawCurtisPointSet,
    ChebyshevRoots,
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
def cheb_roots_num7():
    """the first 7 roots"""
    return [
        0,
        -1 / numpy.sqrt(2),
        1 / numpy.sqrt(2),
        -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
    ]


def test_clenshaw_initial():
    """test default properties"""
    f = ClenshawCurtisPointSet()
    assert f.natural_domain == [-1, 1]
    assert f.points == []
    assert f.number_points == 0


def test_clenshaw_num3():
    """test number of points is changed to 3"""
    f = ClenshawCurtisPointSet()
    f.number_points = 3
    assert numpy.allclose(f.points, [0, -1, 1], atol=1e-10)
    assert f.number_points == 3


def test_clenshaw_num_increased(clenshaw_curtis_num9):
    """test number of points is increased"""
    f = ClenshawCurtisPointSet()
    f.number_points = 3
    f.number_points = 9
    assert numpy.allclose(f.points, clenshaw_curtis_num9, atol=1e-10)
    assert f.number_points == 9


def test_clenshaw_num_decreased(clenshaw_curtis_num9):
    """test number of points is decreased"""
    f = ClenshawCurtisPointSet()
    f.number_points = 16
    f.number_points = 9
    assert numpy.allclose(f.points, clenshaw_curtis_num9, atol=1e-10)
    assert f.number_points == 9


def test_clenshaw_num_error():
    """test number of points is invalid"""
    f = ClenshawCurtisPointSet()
    with pytest.raises(TypeError):
        f.number_points = "Not a number"
    with pytest.raises(ValueError):
        f.number_points = -1


def test_chebroots_initial():
    """test default properties"""
    f = ChebyshevRoots()
    assert f.natural_domain == [-1, 1]
    assert f.points == []
    assert f.number_points == 0


def test_chebroots_num3():
    """test number of points is changed to 3"""
    f = ChebyshevRoots()
    f.number_points = 3
    assert numpy.allclose(
        f.points, [0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)], atol=1e-10
    )
    assert f.number_points == 3


def test_chebroots_num_increased(cheb_roots_num7):
    """test number of points is increased"""
    f = ChebyshevRoots()
    f.number_points = 3
    f.number_points = 7
    assert numpy.allclose(f.points, cheb_roots_num7, atol=1e-10)
    assert f.number_points == 7


def test_chebroots_num_decreased(cheb_roots_num7):
    """test number of points is decreased"""
    f = ChebyshevRoots()
    f.number_points = 16
    f.number_points = 7
    assert numpy.allclose(f.points, cheb_roots_num7, atol=1e-10)
    assert f.number_points == 7


def test_chebroots_num_error():
    """test number of points is invalid"""
    f = ChebyshevRoots()
    with pytest.raises(TypeError):
        f.number_points = "Not a number"
    with pytest.raises(ValueError):
        f.number_points = -1