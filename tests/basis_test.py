import pytest

import numpy

from smolyay.basis import BasisFunction, ChebyshevFirstKind, BasisFunctionSet, NestedBasisFunctionSet, make_nested_chebyshev_points

@pytest.fixture
def expected_points_2():
    """extrema for exactness = 4"""
    return sorted([0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5)])

@pytest.fixture
def expected_points_3():
    """extrema for n = 8"""
    return sorted([0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5), 
            -(((2**0.5)+1)**0.5)/(2**0.75),-(((2**0.5)-1)**0.5)/(2**0.75),
            (((2**0.5)-1)**0.5)/(2**0.75),(((2**0.5)+1)**0.5)/(2**0.75)])

@pytest.fixture
def expected_points_2_set():
    """extrema for exactness = 4"""
    return [0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5)]

@pytest.fixture
def expected_points_3_set():
    """extrema for n = 8"""
    return [0, -1.0, 1.0, -1/(2**0.5), 1/(2**0.5),
            -(((2**0.5)+1)**0.5)/(2**0.75),-(((2**0.5)-1)**0.5)/(2**0.75),
            (((2**0.5)-1)**0.5)/(2**0.75),(((2**0.5)+1)**0.5)/(2**0.75)]

def test_degree_zero():
    """test degree of zero"""
    test_class = ChebyshevFirstKind(0)
    assert test_class.n == 0
    assert test_class.points == [0]

def test_initial_degree_1():
    """test initial when degree is 1"""
    test_class = ChebyshevFirstKind(1)
    assert test_class.n == 1
    assert test_class.points == [-1,1]

def test_initial_degree_2(expected_points_2):
    """test initial when degree is 2"""
    test_class = ChebyshevFirstKind(4)
    assert test_class.n == 4
    assert numpy.allclose(test_class.points,expected_points_2,atol=1e-10)

def test_increase_degree(expected_points_3):
    """test when max degree is increased"""
    test_class = ChebyshevFirstKind(1)
    a = test_class.points
    test_class.n = 8
    assert test_class.n == 8
    assert numpy.allclose(test_class.points, expected_points_3,atol=1e-10)

def test_decrease_degree(expected_points_3):
    """test when max degree is decreased"""
    test_class = ChebyshevFirstKind(16)
    a = test_class.points
    test_class.n = 8
    assert test_class.n == 8
    assert numpy.allclose(test_class.points,expected_points_3,atol=1e-10)

def test_decrease_increase_degree(expected_points_2):
    """test when max degree is decreased"""
    test_class = ChebyshevFirstKind(1)
    a = test_class.points
    test_class.n = 6
    a = test_class.points
    test_class.n = 4
    assert test_class.n == 4
    assert numpy.allclose(test_class.points,expected_points_2,atol=1e-10)

def test_basis_degree_0():
    """Chebyshev polynomial degree 0 is 1"""
    test_class = ChebyshevFirstKind(0)
    for i in range(0,16):
        assert test_class(i) == 1

def test_basis_degree_1():
    """Chebyshev polynomial degree 1 should return input"""
    test_class = ChebyshevFirstKind(1)
    for i in range(0,16):
        assert test_class(i) == i

def test_basis_input_1():
    """Chebyshev polynomial input 1 returns 1 for any degree n"""
    for i in range(0,16):
        test_class = ChebyshevFirstKind(i)
        assert test_class(1) == 1

@pytest.mark.parametrize("x, n, expected",[(2,6,1351),(-3,9,-3880899),
    (13,4,227137),(8,3,2024),(4,9,58106404),(14,5,8550374),(0,12,1),
    (7,5,262087),(-2,4,97),(3,9,3880899),(9,4,51841),(2,11,978122),
    (4,7,937444),(-3,2,17),(6,4,10081),(2,8,18817)])
def test_basis_random_points(x,n,expected):
    """Test chebyshev polynomial at some degree at some input"""
    test_class = ChebyshevFirstKind(n)
    assert test_class(x) == expected

def test_is_basis_abstract():
    """Check BasisFunction is an abstract class"""
    with pytest.raises(TypeError):
        test_class = BasisFunction()

def test_set_initialize_empty():
    """Check NestedBasisFunctionSet initializes with empty set"""
    levels = []
    basis_set = []
    points = []
    test_class = NestedBasisFunctionSet(points,basis_set,levels)
    assert test_class.all_points == []
    assert test_class.levels == []
    assert test_class.basis_set == []

def test_set_initialize_0():
    """Check NestedBasisFunctionSet correctly initializes"""
    levels = [[0]]
    points = [0]
    test_class = NestedBasisFunctionSet(points,[ChebyshevFirstKind(0)],levels)
    assert test_class.all_points == [0]
    assert test_class.levels == [[0]]

def test_set_change(expected_points_2_set):
    """Check NestedBasisFunctionSet updates points when sample_flag changes"""
    levels = [[0]]
    points = [0]
    test_class = NestedBasisFunctionSet(points,[ChebyshevFirstKind(0)],levels)
    basis_set = []
    for i in range(0,9):
        basis_set.append(ChebyshevFirstKind(i))
    test_class.basis_set = basis_set
    test_class.levels = [[0],[1,2],[3,4]]
    test_class.all_points = expected_points_2_set
    assert test_class.all_points == expected_points_2_set
    assert test_class.levels == [[0],[1,2],[3,4]]
    assert test_class.basis_set == basis_set

def test_set_compute_function(expected_points_3_set):
    """Check make_nested_chebyshev_points creates NestedBasisFunctionSet"""
    test_class = make_nested_chebyshev_points(3,ChebyshevFirstKind)
    assert numpy.allclose(
            test_class.all_points,expected_points_3_set,atol=1e-10)
    assert test_class.levels == [[0],[1,2],[3,4],[5,6,7,8]]
    basis_set = test_class.basis_set
    assert len(basis_set) == 9
    for i in range(0,len(basis_set)):
        assert basis_set[i].n == i

def test_set_compute_empty():
    """Check make_nested_chebyshev_points makes empty NestedBasisFunctionSet"""
    test_class = make_nested_chebyshev_points(0,ChebyshevFirstKind)
    assert test_class.all_points == [0]
    assert test_class.levels == [[0]]
    basis_set = test_class.basis_set
    assert len(basis_set) == 1
    assert basis_set[0].n == 0

