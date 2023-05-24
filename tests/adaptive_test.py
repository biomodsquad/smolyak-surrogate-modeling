import numpy
import pytest

from smolyay.adaptive import make_slow_nested_set
from smolyay.basis import (ChebyshevFirstKind, NestedBasisFunctionSet)


@pytest.fixture
def expected_points_3_set():
    """extrema for exactness = 8"""
    return [0, -1.0, 1.0, -1/numpy.sqrt(2), 1/numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75)]

def test_initial_zero():
    """Creates a NestedBasisFunctionSet at exactness 0"""
    f = make_slow_nested_set(0)
    assert f.points == [0]
    assert f.levels == [[0]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 1
    assert basis_functions[0].n == 0


def test_initial_three(expected_points_3_set):
    """Creates a NestedBasisFunctionSet at exactness 3"""
    f = make_slow_nested_set(3)
    assert numpy.allclose(
            f.points,expected_points_3_set,atol=1e-10)
    assert f.levels == [[0],[1,2],[3,4],[5,6,7,8]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 9
    for i in range(0,len(basis_functions)):
        assert basis_functions[i].n == i

def test_initial_four(expected_points_3_set):
    """Creates a NestedBasisFunctionSet at exactness 4"""
    f = make_slow_nested_set(4)
    assert numpy.allclose(
            f.points,expected_points_3_set,atol=1e-10)
    assert f.levels == [[0],[1,2],[3,4],[5,6,7,8],[]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 9
    for i in range(0,len(basis_functions)):
        assert basis_functions[i].n == i
