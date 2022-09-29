import pytest

import numpy
from scipy import special

from smolyay.basis import BasisFunction, ChebyshevFirstKind, BasisFunctionSet, NestedBasisFunctionSet

@pytest.fixture
def expected_points_4():
    """extrema for n = 4"""
    return [-1, -1/numpy.sqrt(2), 0, 1/numpy.sqrt(2), 1]

@pytest.fixture
def expected_points_8():
    """extrema for n = 8"""
    return [-1, -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75), -1/numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 0, 
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 1/numpy.sqrt(2),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75), 1]

@pytest.fixture
def expected_points_2_set():
    """extrema for exactness = 4"""
    return [0, -1.0, 1.0, -1/numpy.sqrt(2), 1/numpy.sqrt(2)]

@pytest.fixture
def expected_points_3_set():
    """extrema for exactness = 8"""
    return [0, -1.0, 1.0, -1/numpy.sqrt(2), 1/numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75)]

def test_cheb_initial_zero():
    """test degree of zero"""
    f = ChebyshevFirstKind(0)
    assert f.n == 0

def test_cheb_initial_1():
    """test initial when degree is 1"""
    f = ChebyshevFirstKind(1)
    assert f.n == 1
    assert f.points == [-1, 1]

def test_cheb_initial_2():
    """test initial when degree is 2"""
    f = ChebyshevFirstKind(2)
    assert f.n == 2
    assert numpy.allclose(f.points,[-1, 0, 1],atol=1e-10)

def test_cheb_initial_4(expected_points_4):
    """test initial when degree is 4"""
    f = ChebyshevFirstKind(4)
    assert f.n == 4
    assert numpy.allclose(f.points,expected_points_4,atol=1e-10)

def test_cheb_initial_8(expected_points_8):
    """test initial when degree is 8"""
    f = ChebyshevFirstKind(8)
    assert f.n == 8
    assert numpy.allclose(f.points,expected_points_8,atol=1e-10)

def test_cheb_call_degree_0():
    """Chebyshev polynomial degree 0 is 1 for all inputs"""
    f = ChebyshevFirstKind(0)
    for i in [-1, -0.5, 0, 0.5, 1]:
        assert f(i) == 1

def test_cheb_call_degree_1():
    """Chebyshev polynomial degree 1 should return input"""
    f = ChebyshevFirstKind(1)
    for i in [-1, -0.5, 0, 0.5, 1]:
        assert f(i) == i

def test_cheb_call_random_points():
    """Test chebyshev polynomial at some degree at some input"""
    numpy.random.seed(567)
    n = numpy.random.randint(20,size = 20)
    x = numpy.random.rand(20) * 2 - 1
    for i in range(20):
        f = ChebyshevFirstKind(n[i])
        assert numpy.isclose(f(x[i]),special.eval_chebyt(n[i],x[i]))

def test_set_initialize_empty():
    """Check BasisFunctionSet initializes with empty set"""
    basis_functions = []
    points = []
    f = BasisFunctionSet(points,basis_functions)
    assert f.points == []
    assert f.basis_functions == []

def test_set_initialize_0():
    """Check NestedBasisFunctionSet correctly initializes"""
    basis_functions = [ChebyshevFirstKind(0)]
    points = [0]
    f = BasisFunctionSet(points,basis_functions)
    assert f.points == [0]
    assert f.basis_functions[0].n == 0

def test_set_invalid_input():
    """Check BasisFunctionSet gives error for invalid inputs"""
    basis_functions = [ChebyshevFirstKind(0)]
    points = [0,1,2]
    with pytest.raises(IndexError):
        f = BasisFunctionSet(points,basis_functions)

def test_set_nested_initialize_empty():
    """Check NestedBasisFunctionSet initializes with empty set"""
    levels = []
    basis_functions = []
    points = []
    f = NestedBasisFunctionSet(points,basis_functions,levels)
    assert f.points == []
    assert f.levels == []
    assert f.basis_functions == []

def test_set_nested_initialize_0():
    """Check NestedBasisFunctionSet correctly initializes"""
    levels = [[0]]
    points = [0]
    f = NestedBasisFunctionSet(points,[ChebyshevFirstKind(0)],levels)
    assert f.points == [0]
    assert f.levels == [[0]]

def test_set_nested_change_levels():
    """Check NestedBasisFunctionSet updates points when level changes"""
    levels = [[0]]
    points = [0]
    f = NestedBasisFunctionSet(points,[ChebyshevFirstKind(0)],levels)
    f.levels = [[0],[1,2],[3,4]]
    assert f.levels == [[0],[1,2],[3,4]]

def test_make_nested_function(expected_points_3_set):
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f = ChebyshevFirstKind.make_nested_set(3)
    assert numpy.allclose(
            f.points,expected_points_3_set,atol=1e-10)
    assert f.levels == [[0],[1,2],[3,4],[5,6,7,8]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 9
    for i in range(0,len(basis_functions)):
        assert basis_functions[i].n == i

def test_make_nested_empty():
    """Check make_nested_set makes empty NestedBasisFunctionSet"""
    f = ChebyshevFirstKind.make_nested_set(0)
    assert f.points == [0]
    assert f.levels == [[0]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 1
    assert basis_functions[0].n == 0


