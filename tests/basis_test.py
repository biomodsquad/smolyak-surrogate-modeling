import pytest


import numpy

from smolyay.basis.basis import BasisFunction, ChebyshevFirstKind

@pytest.fixture
def expected_extrema_2():
    """extrema for exactness = 2"""
    return [0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5)]

@pytest.fixture
def expected_extrema_3():
    """extrema for exactness = 3"""
    return [0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5), 
            -(((2**0.5)+1)**0.5)/(2**0.75),-(((2**0.5)-1)**0.5)/(2**0.75),
            (((2**0.5)-1)**0.5)/(2**0.75),(((2**0.5)+1)**0.5)/(2**0.75)]

def test_exactness_zero():
    """test exactness of zero"""
    test_class = ChebyshevFirstKind(0)
    assert test_class.max_exactness == 0
    assert test_class.points == [0]
    assert test_class.levels == [[0]]

def test_initial_exactness_1():
    """test initial when exactness is 1"""
    test_class = ChebyshevFirstKind(1)
    assert test_class.max_exactness == 1
    assert test_class.points == [0,-1,1]
    assert test_class.levels == [[0],[1,2]]

def test_initial_exactness_2(expected_extrema_2):
    """test initial when exactness is 2"""
    test_class = ChebyshevFirstKind(2)
    assert test_class.max_exactness == 2
    assert numpy.allclose(test_class.points,expected_extrema_2,atol=1e-10)
    assert test_class.levels == [[0],[1,2],[3,4]]

def test_increase_exactness(expected_extrema_3):
    """test when max exactness is increased"""
    test_class = ChebyshevFirstKind(1)
    test_class.max_exactness = 3
    assert test_class.max_exactness == 3
    assert numpy.allclose(test_class.points, expected_extrema_3,atol=1e-10)
    assert test_class.levels == [[0],[1,2],[3,4],[5,6,7,8]]

def test_decrease_exactness(expected_extrema_3):
    """test when max exactness is decreased"""
    test_class = ChebyshevFirstKind(5)
    test_class.max_exactness = 3
    assert test_class.max_exactness == 3
    assert numpy.allclose(test_class.points,expected_extrema_3,atol=1e-10)
    assert test_class.levels == [[0],[1,2],[3,4],[5,6,7,8]]

def test_decrease_increase_exactness(expected_extrema_2):
    """test when max exactness is decreased"""
    test_class = ChebyshevFirstKind(1)
    test_class.max_exactness = 6
    test_class.max_exactness = 2
    assert test_class.max_exactness == 2
    assert numpy.allclose(test_class.points,expected_extrema_2,atol=1e-10)
    assert test_class.levels == [[0],[1,2],[3,4]]

def test_basis_degree_0():
    """Chebyshev polynomial degree 0 is 1"""
    test_class = ChebyshevFirstKind(0)
    for i in range(0,16):
        assert test_class.basis(i,0) == 1

def test_basis_degree_1():
    """Chebyshev polynomial degree 1 should return input"""
    test_class = ChebyshevFirstKind(0)
    for i in range(0,16):
        assert test_class.basis(i,1) == i

def test_basis_input_1():
    """Chebyshev polynomial input 1 returns 1 for any degree n"""
    test_class = ChebyshevFirstKind(0)
    for i in range(0,16):
        assert test_class.basis(1,i) == 1

@pytest.mark.parametrize("x, n, expected",[(2,6,1351),(-3,9,-3880899),
    (13,4,227137),(8,3,2024),(4,9,58106404),(14,5,8550374),(0,12,1),
    (7,5,262087),(-2,4,97),(3,9,3880899),(9,4,51841),(2,11,978122),
    (4,7,937444),(-3,2,17),(6,4,10081),(2,8,18817)])
def test_basis_random_points(x,n,expected):
    """Test chebyshev polynomial at some degree at some input"""
    test_class = ChebyshevFirstKind(0)
    assert test_class.basis(x,n) == expected

def test_is_abstract():
    """Check BasisFunction is an abstract class"""
    with pytest.raises(TypeError):
        test_class = BasisFunction(1)


