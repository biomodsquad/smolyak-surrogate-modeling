import pytest

import numpy
from scipy import special

from smolyay.basis import (BasisFunction, ChebyshevFirstKind,
        ChebyshevSecondKind, BasisFunctionSet, NestedBasisFunctionSet)

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


def test_cheb_initial_zero():
    """test degree of zero"""
    f = ChebyshevFirstKind(0)
    assert f.n == 0
    assert f.points == [0]

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

def test_cheb_initial_4():
    """test initial when degree is 4"""
    expected_points = [-1, -1/numpy.sqrt(2), 0, 1/numpy.sqrt(2), 1]
    f = ChebyshevFirstKind(4)
    assert f.n == 4
    assert numpy.allclose(f.points,expected_points,atol=1e-10)

def test_cheb_initial_8():
    """test initial when degree is 8"""
    expected_points = [-1, -numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -1/numpy.sqrt(2), -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 0,
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 1/numpy.sqrt(2),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75), 1]
    f = ChebyshevFirstKind(8)
    assert f.n == 8
    assert numpy.allclose(f.points,expected_points,atol=1e-10)

def test_cheb_call_degree_0_1():
    """Chebyshev polynomial degree 0 is always 1 and degree 1 returns input"""
    f0 = ChebyshevFirstKind(0)
    f1 = ChebyshevFirstKind(1)
    for i in [-1, -0.5, 0, 0.5, 1]:
        assert f0(i) == 1
        assert f1(i) == i

def test_cheb_call_random_points():
    """Test chebyshev polynomial at some degree at some input"""
    numpy.random.seed(567)
    ns = numpy.random.randint(20,size = 20)
    xs = numpy.random.rand(20) * 2 - 1
    for n,x in zip(ns,xs):
        f = ChebyshevFirstKind(n)
        assert numpy.isclose(f(x),special.eval_chebyt(n,x))

def test_cheb_call_random_points_multi_input():
    """Test that chebyshev polynomial call handles multiple x inputs"""
    numpy.random.seed(567)
    xs = numpy.random.rand(20) * 2 - 1
    f0 = ChebyshevFirstKind(0)
    f1 = ChebyshevFirstKind(1)
    fn = ChebyshevFirstKind(5)
    assert numpy.allclose(f0(xs),special.eval_chebyt(0,xs))
    assert numpy.allclose(f1(xs),special.eval_chebyt(1,xs))
    assert numpy.allclose(fn(xs),special.eval_chebyt(5,xs))


def test_cheb_derivative():
    """Test if the correct derivative is generated."""
    f0 = ChebyshevFirstKind(0)
    f1 = ChebyshevFirstKind(1)
    f2 = ChebyshevFirstKind(2)
    assert f0.derivative(1) == pytest.approx(0)
    assert f1.derivative(1) == pytest.approx(1)
    assert f2.derivative(1) == pytest.approx(4)
    
    assert f0.derivative(-0.5) == pytest.approx(0)
    assert f1.derivative(-0.5) == pytest.approx(1)
    assert f2.derivative(-0.5) == pytest.approx(-2)

    assert numpy.isclose(f0.derivative([1,-0.5]),[0,0]).all()
    assert numpy.isclose(f1.derivative([1,-0.5]),[1,1]).all()
    assert numpy.isclose(f2.derivative([1,-0.5]),[4,-2]).all()


def test_cheb_2nd_derivative():
    """Test if the correct derivative is generated."""
    u0 = ChebyshevSecondKind(0)
    u1 = ChebyshevSecondKind(1)
    u2 = ChebyshevSecondKind(2)
    assert u0.derivative(1) == pytest.approx(0)
    assert u1.derivative(1) == pytest.approx(2)
    assert u2.derivative(1) == pytest.approx(8)

    assert u0.derivative(-1) == pytest.approx(0)
    assert u1.derivative(-1) == pytest.approx(2)
    assert u2.derivative(-1) == pytest.approx(-8)

    assert u0.derivative(0.5) == pytest.approx(0)
    assert u1.derivative(0.5) == pytest.approx(2)
    assert u2.derivative(0.5) == pytest.approx(4)

    assert numpy.isclose(u0.derivative([1,-1,0.5]),[0,0,0]).all()
    assert numpy.isclose(u1.derivative([1,-1,0.5]),[2,2,2]).all()
    assert numpy.isclose(u2.derivative([1,-1,0.5]),[8,-8,4]).all()



def test_cheb_call_invalid_input():
    """Test call raises error if input is outside domain [-1,1]"""
    f = ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5,0.7,3,0.8])


def test_cheb_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1,1]"""
    f = ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)


def test_cheb_2nd_initial_zero():
    """test degree of zero"""
    f = ChebyshevSecondKind(0)
    assert f.n == 0
    assert f.points == [0]


def test_cheb_2nd_initial_1():
    """test initial when degree is 1"""
    f = ChebyshevSecondKind(1)
    assert f.n == 1
    assert f.points == [0]

def test_cheb_2nd_initial_2():
    """test initial when degree is 2"""
    f = ChebyshevSecondKind(2)
    assert f.n == 2
    assert numpy.allclose(f.points,[-0.5, 0.5],atol=1e-10)

def test_cheb_2nd_initial_3():
    """test initial when degree is 3"""
    expected_points = [-1/numpy.sqrt(2), 0, 1/numpy.sqrt(2)]
    f = ChebyshevSecondKind(3)
    assert f.n == 3
    assert numpy.allclose(f.points,expected_points,atol=1e-10)

def test_cheb_2nd_initial_7():
    """test initial when degree is 7"""
    expected_points = [-numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75),
            -1/numpy.sqrt(2), -numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 0,
            numpy.sqrt(numpy.sqrt(2)-1)/(2**0.75), 1/numpy.sqrt(2),
            numpy.sqrt(numpy.sqrt(2)+1)/(2**0.75)]
    f = ChebyshevSecondKind(7)
    assert f.n == 7
    assert numpy.allclose(f.points,expected_points,atol=1e-10)

def test_cheb_2nd_call_degree_0_1():
    """Chebyshev polynomial degree 0 is always 1 and degree 1 returns 2*input"""
    f0 = ChebyshevSecondKind(0)
    f1 = ChebyshevSecondKind(1)
    for i in [-1, -0.5, 0, 0.5, 1]:
        assert f0(i) == 1
        assert f1(i) == i*2

def test_cheb_2nd_call_random_points():
    """Test chebyshev polynomial at some degree at some input"""
    numpy.random.seed(567)
    ns = numpy.random.randint(20,size = 20)
    xs = numpy.random.rand(20) * 2 - 1
    for n,x in zip(ns,xs):
        f = ChebyshevSecondKind(n)
        assert numpy.isclose(f(x),special.eval_chebyu(n,x))

def test_cheb_2nd_call_random_points_multi_input():
    """Test that chebyshev polynomial call handles multiple x inputs"""
    numpy.random.seed(567)
    xs = numpy.random.rand(20) * 2 - 1
    f0 = ChebyshevSecondKind(0)
    f1 = ChebyshevSecondKind(1)
    fn = ChebyshevSecondKind(5)
    assert numpy.allclose(f0(xs),special.eval_chebyu(0,xs))
    assert numpy.allclose(f1(xs),special.eval_chebyu(1,xs))
    assert numpy.allclose(fn(xs),special.eval_chebyu(5,xs))


def test_cheb_2nd_call_invalid_input():
    """Test call raises error if input is outside domain [-1,1]"""
    f = ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5,0.7,3,0.8])


def test_cheb_2nd_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1,1]"""
    f = ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)
    with pytest.raises(ValueError):
        f.derivative([0.5,0.7,3,0.8])


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

def test_set_nested_change_levels_invalid():
    """Check NestedBasisFunctionSet does not allow more levels than points"""
    levels = [[0]]
    points = [0]
    f = NestedBasisFunctionSet(points,[ChebyshevFirstKind(0)],levels)
    with pytest.raises(IndexError):
        f.levels = [[0],[1,2],[3,4]]

def test_set_nested_change_levels(expected_1st_kind):
    """Check NestedBasisFunctionSet updates levels correctly"""
    levels = [[0],[1,2],[3,4],[5,6,7,8]]
    basis_functions = [ChebyshevFirstKind(i) for i in range(9)]
    f = NestedBasisFunctionSet(expected_1st_kind,basis_functions,levels)
    f.levels = [[0],[1,2],[3,4]]
    assert f.levels == [[0],[1,2],[3,4]]

def test_make_nested_function(expected_1st_kind):
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f = ChebyshevFirstKind.make_nested_set(3)
    assert numpy.allclose(
            f.points,expected_1st_kind,atol=1e-10)
    assert f.levels == [[0],[1,2],[3,4],[5,6,7,8]]
    assert len(f.basis_functions) == 9
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_make_nested_empty():
    """Check make_nested_set makes empty NestedBasisFunctionSet"""
    f = ChebyshevFirstKind.make_nested_set(0)
    assert f.points == [0]
    assert f.levels == [[0]]
    assert len(f.basis_functions) == 1
    assert f.basis_functions[0].n == 0

def test_make_nested_function_cheb_2nd(expected_2nd_kind):
    """Check make_nested_set creates NestedBasisFunctionSet"""
    f = ChebyshevSecondKind.make_nested_set(2)
    assert numpy.allclose(f.points,expected_2nd_kind,atol=1e-10)
    assert f.levels == [[0,1],[2,3],[4,5,6,7]]
    assert len(f.basis_functions) == 8
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_make_nested_empty_cheb_2nd():
    """Check make_nested_set makes empty NestedBasisFunctionSet"""
    f = ChebyshevSecondKind.make_nested_set(0)
    assert f.points == [0,0]
    assert f.levels == [[0,1]]
    assert len(f.basis_functions) == 2
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_slow_initial_zero_cheb_1st():
    """Creates a slow NestedBasisFunctionSet at exactness 0"""
    f = ChebyshevFirstKind.make_slow_nested_set(0)
    assert f.points == [0]
    assert f.levels == [[0]]
    assert len(f.basis_functions) == 1
    assert f.basis_functions[0].n == 0


def test_slow_initial_three_cheb_1st(expected_1st_kind):
    """Creates a slow NestedBasisFunctionSet at exactness 3"""
    f = ChebyshevFirstKind.make_slow_nested_set(3)
    assert numpy.allclose(
            f.points,expected_1st_kind,atol=1e-10)
    assert f.levels == [[0], [1, 2], [3, 4], [5, 6, 7, 8]]
    assert len(f.basis_functions) == 9
    for i in range(0,len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_slow_initial_four_cheb_1st(expected_1st_kind):
    """Creates a slow NestedBasisFunctionSet at exactness 4"""
    f = ChebyshevFirstKind.make_slow_nested_set(4)
    assert numpy.allclose(
            f.points,expected_1st_kind,atol=1e-10)
    assert f.levels == [[0], [1, 2], [3, 4], [5, 6, 7, 8], []]
    assert len(f.basis_functions) == 9
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i

def test_slow_custom_rule_cheb_1st():
    """Creates a slower NestedBasisFunctionSet"""
    f = ChebyshevFirstKind.make_slow_nested_set(4, lambda x : x+1)
    assert f.levels == [[0], [1, 2], [], [3, 4], []]
    assert len(f.basis_functions) == 5
    assert numpy.allclose(f.points,
                          [0, -1, 1, -1/numpy.sqrt(2), 1/numpy.sqrt(2)])


def test_slow_nested_zero_cheb_2nd():
    """Check make_slow_nested_set makes empty NestedBasisFunctionSet"""
    f = ChebyshevSecondKind.make_slow_nested_set(0)
    assert f.points == [0, 0]
    assert f.levels == [[0, 1]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 2
    for i in range(len(f.basis_functions)):
        assert f.basis_functions[i].n == i


def test_slow_nested_two_cheb_2nd(expected_2nd_kind):
    """Check make_slow_nested_set creates NestedBasisFunctionSet"""
    f = ChebyshevSecondKind.make_slow_nested_set(2)
    assert numpy.allclose(f.points,expected_2nd_kind,atol=1e-10)
    assert f.levels == [[0,1],[2,3],[4,5,6,7]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 8
    for i in range(len(basis_functions)):
        assert basis_functions[i].n == i

def test_slow_nested_three_cheb_2nd(expected_2nd_kind):
    """Check make_slow_nested_set creates NestedBasisFunctionSet"""
    f = ChebyshevSecondKind.make_slow_nested_set(3)
    assert numpy.allclose(f.points,expected_2nd_kind,atol=1e-10)
    assert f.levels == [[0,1],[2,3],[4,5,6,7],[]]
    basis_functions = f.basis_functions
    assert len(basis_functions) == 8
    for i in range(0,len(basis_functions)):
        assert basis_functions[i].n == i

def test_slow_nested_custom_rule_cheb_2nd():
    """Creates a slower NestedBasisFunctionSet"""
    f1 = ChebyshevSecondKind.make_slow_nested_set(2,lambda x : x + 1)
    assert f1.levels == [[0, 1], [], [2, 3]]
    assert numpy.allclose(f1.points,
                          [0, 0, -1/numpy.sqrt(2), 1/numpy.sqrt(2)])
    assert len(f1.basis_functions) == 4
    for i in range(len(f1.basis_functions)):
        assert f1.basis_functions[i].n == i


