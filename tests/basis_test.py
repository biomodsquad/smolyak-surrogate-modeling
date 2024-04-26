import pytest

import numpy
from scipy import special

import smolyay


def test_cheb_initial():
    """test degrees returns correctly"""
    f2 = smolyay.basis.ChebyshevFirstKind(2)
    assert f2.degree == 2
    assert isinstance(f2.degree, int)
    assert numpy.array_equal(f2.domain, [-1, 1])
    f2.degree = 3
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


@pytest.mark.parametrize("degree", list(range(3)))
def test_cheb_call(degree):
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    numpy.random.seed(567)
    xs = numpy.linspace(-1, 1, 20)
    for x in xs:
        assert f(x) == pytest.approx(special.eval_chebyt(degree, x))


@pytest.mark.parametrize("degree", list(range(3)))
def test_cheb_call_random_points_multi_input(degree):
    """Test chebyshev polynomial call handles inputs with complex shape"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    numpy.random.seed(567)
    xs = numpy.random.rand(20, 3, 4, 6) * 2 - 1
    answer = f(xs)
    answer_check = special.eval_chebyt(degree, xs)
    assert answer.shape == answer_check.shape
    assert numpy.allclose(answer, answer_check)


def test_cheb_call_extrema_points():
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    extrema_points = [-1.0, -1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2), 1]
    extrema_output = [1, -1, 1, -1, 1]
    assert numpy.allclose(f(extrema_points), extrema_output)


def test_cheb_call_root_points():
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    root_points = [
        -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
    ]
    assert numpy.allclose(f(root_points), numpy.zeros(4))


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [0, 0]),
        (1, [1, 1]),
        (2, [4, -2]),
    ],
)
def test_cheb_derivative(degree, answer):
    """Test if the correct derivative is generated."""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    assert f.derivative(1) == pytest.approx(answer[0])
    assert f.derivative(-0.5) == pytest.approx(answer[1])
    assert numpy.allclose(f.derivative([1, -0.5]), answer)


def test_cheb_call_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, 0.8])
    with pytest.raises(ValueError):
        f([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_cheb_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)
    with pytest.raises(ValueError):
        f.derivative([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_cheb_2nd_initial():
    """test degrees returns correctly"""
    f2 = smolyay.basis.ChebyshevSecondKind(2)
    assert f2.degree == 2
    assert isinstance(f2.degree, int)
    assert numpy.array_equal(f2.domain, [-1, 1])
    f2.degree = 3
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


@pytest.mark.parametrize("degree", list(range(5)))
def test_cheb_2nd_call(degree):
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevSecondKind(degree)
    numpy.random.seed(567)
    xs = numpy.linspace(-1, 1, 20)
    for x in xs:
        assert f(x) == pytest.approx(special.eval_chebyu(degree, x))


@pytest.mark.parametrize("degree", list(range(3)))
def test_cheb_2nd_call_random_points_multi_input(degree):
    """Test chebyshev polynomial call handles inputs with complex shape"""
    f = smolyay.basis.ChebyshevSecondKind(degree)
    numpy.random.seed(567)
    xs = numpy.random.rand(20, 3, 4, 6) * 2 - 1
    answer = f(xs)
    answer_check = special.eval_chebyu(degree, xs)
    assert answer.shape == answer_check.shape
    assert numpy.allclose(answer, answer_check)


def test_cheb_2nd_call_root_points():
    """Test chebyshev polynomial roots"""
    f = smolyay.basis.ChebyshevSecondKind(3)
    root_points = [-1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2)]
    assert numpy.allclose(f(root_points), numpy.zeros(3))


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [0, 0, 0]),
        (1, [2, 2, 2]),
        (2, [8, -8, 4]),
    ],
)
def test_cheb_2nd_derivative(degree, answer):
    """Test if the correct derivative is generated."""
    u = smolyay.basis.ChebyshevSecondKind(degree)
    assert u.derivative(1) == pytest.approx(answer[0])
    assert u.derivative(-1) == pytest.approx(answer[1])
    assert u.derivative(0.5) == pytest.approx(answer[2])
    assert numpy.allclose(u.derivative([1, -1, 0.5]), answer)


def test_cheb_2nd_call_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = smolyay.basis.ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, 0.8])
    with pytest.raises(ValueError):
        f([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_cheb_2nd_derivative_invalid_input():
    """Test call raises error if input is outside domain [-1, 1]"""
    f = smolyay.basis.ChebyshevSecondKind(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)
    with pytest.raises(ValueError):
        f.derivative([0.5, 0.7, 3, 0.8])
    with pytest.raises(ValueError):
        f.derivative([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_trig_initial():
    """test degrees returns correctly"""
    f2 = smolyay.basis.Trigonometric(2)
    assert f2.frequency == 2
    assert isinstance(f2.frequency, int)
    assert numpy.array_equal(f2.domain, [0, 2 * numpy.pi])
    f2.frequency = 3
    assert f2.frequency == 3
    assert isinstance(f2.frequency, int)


@pytest.mark.parametrize(
    "frequency,c",
    [
        (0, 0),
        (1, 1j),
        (-1, 1j * -1),
    ],
)
def test_trig_call(frequency, c):
    """Test call method of trigonometric basis function"""
    f = smolyay.basis.Trigonometric(frequency)
    for i in [0, numpy.pi / 3, 3 * numpy.pi / 2]:
        assert f(i) == numpy.exp(i * c)


@pytest.mark.parametrize(
    "frequency,c",
    [
        (0, 0),
        (1, 1j),
        (3, 1j * 3),
    ],
)
def test_trig_call_random_points_multi_input(frequency, c):
    """Test that Trigonometric polynomial call handles multiple x inputs"""
    f = smolyay.basis.Trigonometric(frequency)
    numpy.random.seed(567)
    xs = numpy.random.rand(20, 6, 3, 2) * 2 * numpy.pi
    assert numpy.allclose(f(xs), numpy.exp(xs * c))


@pytest.mark.parametrize(
    "frequency,answer",
    [
        (0, [0, 0]),
        (1, [1j * numpy.exp(1j), 1j * numpy.exp(numpy.pi * 1j * 1 / 6)]),
        (
            -1,
            [-1j * numpy.exp(1 * 1j * (-1)), -1j * numpy.exp(numpy.pi * 1j * (-1) / 6)],
        ),
    ],
)
def test_trig_derivative(frequency, answer):
    """Test if the correct derivative is generated."""
    f = smolyay.basis.Trigonometric(frequency)
    assert f.derivative(1) == pytest.approx(answer[0])
    assert f.derivative(numpy.pi / 6) == pytest.approx(answer[1])
    assert numpy.allclose(f.derivative([1, numpy.pi / 6]), answer)


def test_trig_call_invalid_input():
    """Test call raises error if input is outside domain [0, 2pi]"""
    f = smolyay.basis.Trigonometric(4)
    with pytest.raises(ValueError):
        f(6.5)
    with pytest.raises(ValueError):
        f(-1)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, -0.8])
    with pytest.raises(ValueError):
        f([[0.5, 0.7, -3, 0.8], [0.5, 0.7, 0.8, 0.5]])


def test_trig_derivative_invalid_input():
    """Test call raises error if input is outside domain [0, 2pi]"""
    f = smolyay.basis.Trigonometric(4)
    with pytest.raises(ValueError):
        f.derivative(-0.04)
    with pytest.raises(ValueError):
        f.derivative(6.5)
    with pytest.raises(ValueError):
        f.derivative([0.5, 0.7, 3, -0.8])
    with pytest.raises(ValueError):
        f.derivative([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_set_initialize():
    """Check BasisFunctionSet correctly initializes"""
    f = smolyay.basis.BasisFunctionSet([])
    f2 = smolyay.basis.BasisFunctionSet([smolyay.basis.ChebyshevFirstKind(0)])
    assert f.basis_functions == []
    assert f2.basis_functions[0].degree == 0
