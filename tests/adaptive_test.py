import numpy
import pytest

from smolyay.adaptive import (make_slow_nested_set, make_slow_nested_set_2)
from smolyay.basis import (ChebyshevFirstKind, NestedBasisFunctionSet)


@pytest.fixture
def expected_1st_kind():
    """extrema for exactness = 8"""
    return [0, -1.0, 1.0, -1/numpy.sqrt(2), 1/numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75)]

@pytest.fixture
def expected_2nd_kind():
    """roots for 15th order Chebyshev polynomial of the second kind"""
    return [0, 0, -1/numpy.sqrt(2), 1/numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75)]

def test_initial_zero():
    """Creates a NestedBasisFunctionSet at exactness 0"""
    f = make_slow_nested_set(0)
    assert f.points == [0]
    assert f.levels == [[0]]
    assert len(f.basis_functions) == 1
    assert f.basis_functions[0].n == 0


def test_initial_three(expected_1st_kind):
    """Creates a NestedBasisFunctionSet at exactness 3"""
    f = make_slow_nested_set(3)
    assert numpy.allclose(
            f.points,expected_1st_kind,atol=1e-10)
    assert f.levels == [[0], [1, 2], [3, 4], [5, 6, 7, 8]]
    assert len(f.basis_functions) == 9
    for i in range(0,len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_initial_four(expected_1st_kind):
    """Creates a NestedBasisFunctionSet at exactness 4"""
    f = make_slow_nested_set(4)
    assert numpy.allclose(
            f.points,expected_1st_kind,atol=1e-10)
    assert f.levels == [[0], [1, 2], [3, 4], [5, 6, 7, 8], []]
    assert len(f.basis_functions) == 9
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_custom_rule():
    """Creates a slower NestedBasisFunctionSet"""
    f = make_slow_nested_set(4, lambda x : x+1)
    assert f.levels == [[0], [1, 2], [], [3, 4], []]
    assert len(f.basis_functions) == 5
    assert numpy.allclose(f.points,
                          [0, -1, 1, -1/numpy.sqrt(2), 1/numpy.sqrt(2)])


def test_slow_nested_zero_2nd_kind():
    """Check make_nested_set makes empty NestedBasisFunctionSet"""
    f = make_slow_nested_set_2(0)
    assert f.points == [0, 0]
    assert f.levels == [[0, 1]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 2
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i


def test_slow_nested_two_2nd_kind(expected_2nd_kind):
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f = make_slow_nested_set_2(2)
    assert numpy.allclose(f.points,expected_2nd_kind,atol=1e-10)
    assert f.levels == [[0,1],[2,3],[4,5,6,7]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 8
    for i in range(len(basis_functions)):
        assert basis_functions[i].n == i

def test_slow_nested_three_2nd_kind(expected_2nd_kind):
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f = make_slow_nested_set_2(3)
    assert numpy.allclose(f.points,expected_2nd_kind,atol=1e-10)
    assert f.levels == [[0,1],[2,3],[4,5,6,7],[]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 8
    for i in range(0,len(basis_functions)):
        assert basis_functions[i].n == i

def test_slow_nested_custom_rule_2nd_kind():
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f1 = make_slow_nested_set_2(2,lambda x : x + 1)
    assert f1.levels == [[0, 1], [], [2, 3]]
    assert numpy.allclose(f1.points,
                          [0, 0, -1/numpy.sqrt(2), 1/numpy.sqrt(2)])
    assert len(f1.basis_functions) == 4
    for i in range(len(f1.basis_functions)):
        assert f1.basis_functions[i].n == i

