import pytest

import numpy
from scipy import special

from smolyay.basis import (
    BasisFunction,
    ChebyshevFirstKind,
    ChebyshevSecondKind,
    Trigonometric,
    BasisFunctionSet,
)


def test_cheb_initial():
    """test degrees returns correctly"""
    f2 = ChebyshevFirstKind(2)
    assert f2.n == 2
    assert numpy.array_equal(f2.domain,[-1, 1])


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
    ns = numpy.random.randint(20, size=20)
    xs = numpy.random.rand(20) * 2 - 1
    for n, x in zip(ns, xs):
        f = ChebyshevFirstKind(n)
        assert numpy.isclose(f(x), special.eval_chebyt(n, x))


def test_cheb_call_random_points_multi_input():
    """Test that chebyshev polynomial call handles multiple x inputs"""
    numpy.random.seed(567)
    xs = numpy.random.rand(20) * 2 - 1
    f0 = ChebyshevFirstKind(0)
    f1 = ChebyshevFirstKind(1)
    fn = ChebyshevFirstKind(5)
    assert numpy.allclose(f0(xs), special.eval_chebyt(0, xs))
    assert numpy.allclose(f1(xs), special.eval_chebyt(1, xs))
    assert numpy.allclose(fn(xs), special.eval_chebyt(5, xs))


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

    assert numpy.isclose(f0.derivative([1, -0.5]), [0, 0]).all()
    assert numpy.isclose(f1.derivative([1, -0.5]), [1, 1]).all()
    assert numpy.isclose(f2.derivative([1, -0.5]), [4, -2]).all()


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

    assert numpy.isclose(u0.derivative([1, -1, 0.5]), [0, 0, 0]).all()
    assert numpy.isclose(u1.derivative([1, -1, 0.5]), [2, 2, 2]).all()
    assert numpy.isclose(u2.derivative([1, -1, 0.5]), [8, -8, 4]).all()


def test_cheb_call_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, 0.8])


def test_cheb_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)


def test_cheb_2nd_initial():
    """test degrees returns correctly"""
    f2 = ChebyshevSecondKind(2)
    assert f2.n == 2
    assert numpy.array_equal(f2.domain,[-1, 1])


def test_cheb_2nd_call_degree_0_1():
    """Chebyshev polynomial degree 0 is always 1 and degree 1 returns 2*input"""
    f0 = ChebyshevSecondKind(0)
    f1 = ChebyshevSecondKind(1)
    for i in [-1, -0.5, 0, 0.5, 1]:
        assert f0(i) == 1
        assert f1(i) == i * 2


def test_cheb_2nd_call_random_points():
    """Test chebyshev polynomial at some degree at some input"""
    numpy.random.seed(567)
    ns = numpy.random.randint(20, size=20)
    xs = numpy.random.rand(20) * 2 - 1
    for n, x in zip(ns, xs):
        f = ChebyshevSecondKind(n)
        assert numpy.isclose(f(x), special.eval_chebyu(n, x))


def test_cheb_2nd_call_random_points_multi_input():
    """Test that chebyshev polynomial call handles multiple x inputs"""
    numpy.random.seed(567)
    xs = numpy.random.rand(20) * 2 - 1
    f0 = ChebyshevSecondKind(0)
    f1 = ChebyshevSecondKind(1)
    fn = ChebyshevSecondKind(5)
    assert numpy.allclose(f0(xs), special.eval_chebyu(0, xs))
    assert numpy.allclose(f1(xs), special.eval_chebyu(1, xs))
    assert numpy.allclose(fn(xs), special.eval_chebyu(5, xs))


def test_cheb_2nd_call_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, 0.8])


def test_cheb_2nd_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)
    with pytest.raises(ValueError):
        f.derivative([0.5, 0.7, 3, 0.8])


def test_trig_initial():
    """test degrees returns correctly"""
    f2 = Trigonometric(2)
    assert f2.n == 2
    assert numpy.array_equal(f2.domain,[0, 2 * numpy.pi])


def test_trig_call():
    """Test call method of trigonometric basis function"""
    f0 = Trigonometric(0)
    f1 = Trigonometric(1)
    f2 = Trigonometric(2)
    f3 = Trigonometric(3)
    for i in [0, numpy.pi / 3, 3 * numpy.pi / 2]:
        assert f0(i) == numpy.exp(i * 1j * 0)
        assert f1(i) == numpy.exp(i * 1j)
        assert f2(i) == numpy.exp(i * 1j * (-1))
        assert f3(i) == numpy.exp(i * 1j * 2)


def test_trig_call_random_points_multi_input():
    """Test that Trigonometric polynomial call handles multiple x inputs"""
    numpy.random.seed(567)
    xs = numpy.random.rand(20) * 2 * numpy.pi
    f0 = Trigonometric(0)
    f1 = Trigonometric(1)
    fn = Trigonometric(5)
    assert numpy.allclose(f0(xs), numpy.exp(xs * 1j * 0))
    assert numpy.allclose(f1(xs), numpy.exp(xs * 1j))
    assert numpy.allclose(fn(xs), numpy.exp(xs * 1j * 3))


def test_trig_derivative():
    """Test if the correct derivative is generated."""
    f0 = Trigonometric(0)
    f1 = Trigonometric(1)
    f2 = Trigonometric(2)
    assert f0.derivative(1) == pytest.approx(0)
    assert f1.derivative(1) == pytest.approx(1j * numpy.exp(1j))
    assert f2.derivative(1) == pytest.approx(-1j * numpy.exp(1 * 1j * (-1)))

    assert f0.derivative(numpy.pi / 6) == pytest.approx(0)
    assert f1.derivative(numpy.pi / 6) == pytest.approx(
        1j * numpy.exp(numpy.pi * 1j * 1 / 6)
    )
    assert f2.derivative(numpy.pi / 6) == pytest.approx(
        -1j * numpy.exp(numpy.pi * 1j * (-1) / 6)
    )

    assert numpy.isclose(f0.derivative([1, numpy.pi / 6]), [0, 0]).all()
    assert numpy.isclose(
        f1.derivative([1, numpy.pi / 6]),
        [1j * numpy.exp(1j), 1j * numpy.exp(numpy.pi * 1j * 1 / 6)],
    ).all()
    assert numpy.isclose(
        f2.derivative([1, numpy.pi / 6]),
        [-1j * numpy.exp(1 * 1j * (-1)), -1j * numpy.exp(numpy.pi * 1j * (-1) / 6)],
    ).all()


def test_trig_call_invalid_input():
    """Test call raises error if input is outside domain [0, 2pi]"""
    f = Trigonometric(4)
    with pytest.raises(ValueError):
        f(6.5)
    with pytest.raises(ValueError):
        f(-1)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, -0.8])


def test_trig_derivative_invalid_input():
    """Test call raises error if input is outside domain [0, 2pi]"""
    f = Trigonometric(4)
    with pytest.raises(ValueError):
        f.derivative(-0.04)
    with pytest.raises(ValueError):
        f.derivative(6.5)


def test_trig_sigma():
    """Test whether correct frequency is generated given a degree."""
    f7 = Trigonometric(7)
    f10 = Trigonometric(10)
    assert f7.sigma == pytest.approx(4)
    assert f10.sigma == pytest.approx(-5)


def test_set_initialize_empty():
    """Check BasisFunctionSet initializes with empty set"""
    basis_functions = []
    f = BasisFunctionSet(basis_functions)
    assert f.basis_functions == []


def test_set_initialize_0():
    """Check BasisFunctionSet correctly initializes"""
    basis_functions = [ChebyshevFirstKind(0)]
    f = BasisFunctionSet(basis_functions)
    assert f.basis_functions[0].n == 0
