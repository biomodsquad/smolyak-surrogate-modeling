import pytest

import numpy

from smolyay.samples import (
    UnidimensionalPointSet,
    ClenshawCurtisPointSet,
    NestedClenshawCurtisPointSet,
    TrigonometricPointSet,
    NestedTrigonometricPointSet,
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
    f = ClenshawCurtisPointSet(7)
    assert numpy.array_equal(f.domain, [-1, 1])
    assert f.degree == 7


def test_clenshaw_generate_values(clenshaw_curtis_num9):
    """test the output of generate_points"""
    assert numpy.allclose(ClenshawCurtisPointSet(0).points, [0], atol=1e-10)
    assert numpy.allclose(ClenshawCurtisPointSet(1).points, [-1, 1], atol=1e-10)
    assert numpy.allclose(ClenshawCurtisPointSet(2).points, [-1, 0, 1], atol=1e-10)
    assert numpy.allclose(
        ClenshawCurtisPointSet(3).points, [-1, -0.5, 0.5, 1], atol=1e-10
    )
    assert numpy.allclose(
        ClenshawCurtisPointSet(8).points, sorted(clenshaw_curtis_num9), atol=1e-10
    )


def test_nestedclenshaw_initial():
    """test default properties"""
    f = NestedClenshawCurtisPointSet(7)
    assert numpy.array_equal(f.domain, [-1, 1])
    assert f.max_level == 7
    assert f._valid_cache == True
    f.points
    assert f._valid_cache == False


def test_nestedclenshaw_values(clenshaw_curtis_num9):
    """test number of points"""
    assert numpy.allclose(NestedClenshawCurtisPointSet(0).points, [0], atol=1e-10)
    assert numpy.allclose(
        NestedClenshawCurtisPointSet(1).points, [0, -1, 1], atol=1e-10
    )
    assert numpy.allclose(
        NestedClenshawCurtisPointSet(3).points, [clenshaw_curtis_num9], atol=1e-10
    )

def test_nestedclenshaw_growth():
    """test growth order matches point accumulation"""
    f = NestedClenshawCurtisPointSet(7)
    assert f._growth_rule(0) == 1
    assert f._growth_rule(1) == 3
    assert f._growth_rule(2) == 5
    assert f._growth_rule(3) == 9

def test_nestedclenshaw_levels():
    """test number of levels"""
    assert NestedClenshawCurtisPointSet(0).levels == [[0]]
    assert NestedClenshawCurtisPointSet(1).levels == [[0],[1,2]]
    assert NestedClenshawCurtisPointSet(2).levels == [[0],[1,2],[3,4]]
    assert NestedClenshawCurtisPointSet(3).levels == [[0],[1,2],[3,4],[5,6,7,8]]


def test_nestedclenshaw_growth_point_match():
    """test the growth rule matches number of points accumulating"""
    for i in range(10):
        f = NestedClenshawCurtisPointSet(i)
        assert f._growth_rule(i) == len(f.points)
        assert len(f.points) - 1 == f.levels[-1][-1]


def test_trig_initial():
    """test default properties"""
    f = TrigonometricPointSet(7)
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])
    assert f.frequency == 7


def test_trig_generate_values(trig_num9):
    """test number of points is changed to 3"""
    assert numpy.allclose(TrigonometricPointSet(0).points, [0], atol=1e-10)
    assert numpy.allclose(TrigonometricPointSet(1).points, [0], atol=1e-10)
    assert numpy.allclose(TrigonometricPointSet(2).points, [0, numpy.pi], atol=1e-10)
    assert numpy.allclose(
        TrigonometricPointSet(3).points,
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
        atol=1e-10,
    )
    assert numpy.allclose(
        TrigonometricPointSet(9).points, sorted(trig_num9), atol=1e-10
    )


def test_nestedtrig_initial():
    """test default properties"""
    f = NestedTrigonometricPointSet(7)
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])
    assert f.max_level == 7


def test_nestedtrig_generate_values(trig_num9):
    """test number of points is changed to 3"""
    assert numpy.allclose(NestedTrigonometricPointSet(0).points, [0], atol=1e-10)
    assert numpy.allclose(
        NestedTrigonometricPointSet(1).points,
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
        atol=1e-10,
    )
    assert numpy.allclose(NestedTrigonometricPointSet(2).points, trig_num9, atol=1e-10)

def test_nestedtrig_growth():
    """test growth order matches point accumulation"""
    f = NestedTrigonometricPointSet(7)
    assert f._growth_rule(0) == 1
    assert f._growth_rule(1) == 3
    assert f._growth_rule(2) == 9
    assert f._growth_rule(3) == 27

def test_nestedtrig_levels():
    """test number of levels"""
    assert NestedTrigonometricPointSet(0).levels == [[0]]
    assert NestedTrigonometricPointSet(1).levels == [[0],[1,2]]
    assert NestedTrigonometricPointSet(2).levels == [[0],[1,2],[3,4,5,6,7,8]]
    assert NestedTrigonometricPointSet(3).levels == [[0],[1,2],[3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]


def test_nestedtrig_growth_point_match():
    """test the growth rule matches number of points accumulating"""
    for i in range(7):
        f = NestedTrigonometricPointSet(i)
        assert f._growth_rule(i) == len(f.points)
        assert len(f.points) - 1 == f.levels[-1][-1]