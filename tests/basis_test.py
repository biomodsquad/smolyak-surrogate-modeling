import pytest

import numpy
import scipy.special

import smolyay


def test_cheb_initial():
    """test degrees returns correctly"""
    f2 = smolyay.basis.ChebyshevFirstKind(2)
    assert f2.degree == 2
    assert isinstance(f2.degree, int)
    assert numpy.array_equal(f2.domain, [-1, 1])
    f2.degree = float(3)
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [1, 1, 1, 1]),
        (1, [0.5, 1, -1, -0.25]),
        (2, [-0.5, 1, 1, -0.875]),
    ],
)
def test_cheb_call(degree, answer):
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    assert f(0.5) == pytest.approx(answer[0])
    assert f(1) == pytest.approx(answer[1])
    assert f(-1) == pytest.approx(answer[2])
    assert f(-0.25) == pytest.approx(answer[3])


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [1, 1, 1, 1]),
        (1, [0.5, 1, -1, -0.25]),
        (2, [-0.5, 1, 1, -0.875]),
    ],
)
def test_cheb_call_1D(degree, answer):
    """Test chebyshev polynomial call handles inputs from a 1D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [0.5, 1, -1, -0.25]
    assert numpy.shape(f(xs)) == numpy.shape(answer)
    assert numpy.allclose(f(xs), answer)


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [[1, 1, 1, 1], [1, 1, 1, 1]]),
        (1, [[0.5, 1, -1, -0.25], [-0.5, -1, 1, -0.25]]),
        (2, [[-0.5, 1, 1, -0.875], [-0.5, 1, 1, -0.875]]),
    ],
)
def test_cheb_call_2D(degree, answer):
    """Test chebyshev polynomial call handles inputs from a 2D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [[0.5, 1, -1, -0.25], [-0.5, -1, 1, -0.25]]
    xs2 = numpy.reshape(xs[0], (1, 4))
    xs3 = numpy.reshape(xs[0], (4, 1))
    answer2 = numpy.reshape(answer[0], (1, 4))
    answer3 = numpy.reshape(answer[0], (4, 1))
    assert numpy.shape(f(xs)) == numpy.shape(answer)
    assert numpy.allclose(f(xs), answer)
    assert numpy.shape(f(xs2)) == (1, 4)
    assert numpy.allclose(f(xs2), answer2)
    assert numpy.shape(f(xs3)) == (4, 1)
    assert numpy.allclose(f(xs3), answer3)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_cheb_call_3D(degree):
    """Test chebyshev polynomial call handles inputs from a 3D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [
        [[0.5, 1, -1, -0.25], [-0.5, -1, 1, -0.25], [-0.25, 1, -1, 0.5]],
        [[0.5, 1, 1, -0.25], [-0.5, -1, 1, -0.25], [-0.25, 1, 1, 0.5]],
    ]
    xs2 = numpy.reshape(xs[0][0], (1, 1, 4))
    xs3 = numpy.reshape(xs[0][0], (1, 4, 1))
    xs4 = numpy.reshape(xs[0][0], (1, 1, 4))
    xs5 = numpy.reshape(xs[0], (1, 3, 4))
    xs6 = numpy.reshape(xs[0], (3, 4, 1))
    xs7 = numpy.ones((1, 1, 1))
    assert numpy.shape(f(xs)) == numpy.shape(xs)
    assert numpy.allclose(f(xs), scipy.special.eval_chebyt(degree, xs))
    assert numpy.shape(f(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f(xs2), scipy.special.eval_chebyt(degree, xs2))
    assert numpy.shape(f(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f(xs3), scipy.special.eval_chebyt(degree, xs3))
    assert numpy.shape(f(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f(xs4), scipy.special.eval_chebyt(degree, xs4))
    assert numpy.shape(f(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f(xs5), scipy.special.eval_chebyt(degree, xs5))
    assert numpy.shape(f(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f(xs6), scipy.special.eval_chebyt(degree, xs6))
    assert numpy.shape(f(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f(xs7), scipy.special.eval_chebyt(degree, xs7))


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


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [0, 0, 0, 0]),
        (1, [1, 1, 1, 1]),
        (2, [2, 4, -4, -1]),
    ],
)
def test_cheb_derivative(degree, answer):
    """Test chebyshev polynomial derivative at some degree at some input"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    assert f.derivative(0.5) == pytest.approx(answer[0])
    assert f.derivative(1) == pytest.approx(answer[1])
    assert f.derivative(-1) == pytest.approx(answer[2])
    assert f.derivative(-0.25) == pytest.approx(answer[3])


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [0, 0, 0, 0]),
        (1, [1, 1, 1, 1]),
        (2, [2, 4, -4, -1]),
    ],
)
def test_cheb_derivative_1D(degree, answer):
    """Test chebyshev polynomial derivative handles inputs from a 1D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [0.5, 1, -1, -0.25]
    assert numpy.shape(f.derivative(xs)) == numpy.shape(answer)
    assert numpy.allclose(f.derivative(xs), answer)


@pytest.mark.parametrize(
    "degree,answer",
    [
        (0, [[0, 0, 0, 0], [0, 0, 0, 0]]),
        (1, [[1, 1, 1, 1], [1, 1, 1, 1]]),
        (2, [[2, 4, -4, -1], [-2, -4, 4, -1]]),
    ],
)
def test_cheb_derivative_2D(degree, answer):
    """Test chebyshev polynomial derivative handles inputs from a 2D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [[0.5, 1, -1, -0.25], [-0.5, -1, 1, -0.25]]
    xs2 = numpy.reshape(xs[0], (1, 4))
    xs3 = numpy.reshape(xs[0], (4, 1))
    answer2 = numpy.reshape(answer[0], (1, 4))
    answer3 = numpy.reshape(answer[0], (4, 1))
    assert numpy.shape(f.derivative(xs)) == numpy.shape(answer)
    assert numpy.allclose(f.derivative(xs), answer)
    assert numpy.shape(f.derivative(xs2)) == (1, 4)
    assert numpy.allclose(f.derivative(xs2), answer2)
    assert numpy.shape(f.derivative(xs3)) == (4, 1)
    assert numpy.allclose(f.derivative(xs3), answer3)


@pytest.mark.parametrize(
    "degree,answer",
    [
        (
            0,
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ),
        (
            1,
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ],
        ),
        (
            2,
            [
                [[2, 4, -4, -1], [-2, -4, 4, -1], [-1, 4, -4, 2]],
                [[2, 4, 4, -1], [-2, -4, 4, -1], [-1, 4, 4, 2]],
            ],
        ),
    ],
)
def test_cheb_derivative_3D(degree, answer):
    """Test chebyshev polynomial derivative handles inputs from a 3D array"""
    f = smolyay.basis.ChebyshevFirstKind(degree)
    xs = [
        [[0.5, 1, -1, -0.25], [-0.5, -1, 1, -0.25], [-0.25, 1, -1, 0.5]],
        [[0.5, 1, 1, -0.25], [-0.5, -1, 1, -0.25], [-0.25, 1, 1, 0.5]],
    ]
    xs2 = numpy.reshape(xs[0][0], (1, 1, 4))
    xs3 = numpy.reshape(xs[0][0], (1, 4, 1))
    xs4 = numpy.reshape(xs[0][0], (1, 1, 4))
    xs5 = numpy.reshape(xs[0], (1, 3, 4))
    xs6 = numpy.reshape(xs[0], (3, 4, 1))
    xs7 = numpy.ones((1, 1, 1)) * xs[0][0][0]
    answer2 = numpy.reshape(answer[0][0], (1, 1, 4))
    answer3 = numpy.reshape(answer[0][0], (1, 4, 1))
    answer4 = numpy.reshape(answer[0][0], (1, 1, 4))
    answer5 = numpy.reshape(answer[0], (1, 3, 4))
    answer6 = numpy.reshape(answer[0], (3, 4, 1))
    answer7 = numpy.ones((1, 1, 1)) * answer[0][0][0]
    assert numpy.shape(f.derivative(xs)) == numpy.shape(xs)
    assert numpy.allclose(f.derivative(xs), answer)
    assert numpy.shape(f.derivative(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f.derivative(xs2), answer2)
    assert numpy.shape(f.derivative(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f.derivative(xs3), answer3)
    assert numpy.shape(f.derivative(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f.derivative(xs4), answer4)
    assert numpy.shape(f.derivative(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f.derivative(xs5), answer5)
    assert numpy.shape(f.derivative(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f.derivative(xs6), answer6)
    assert numpy.shape(f.derivative(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f.derivative(xs7), answer7)


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
    f2.degree = float(3)
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


@pytest.mark.parametrize("degree", list(range(5)))
def test_cheb_2nd_call(degree):
    """Test chebyshev polynomial at some degree at some input"""
    f = smolyay.basis.ChebyshevSecondKind(degree)
    assert f(0.5) == pytest.approx(scipy.special.eval_chebyu(degree, 0.5))
    assert f(1) == pytest.approx(scipy.special.eval_chebyu(degree, 1))
    assert f(-1) == pytest.approx(scipy.special.eval_chebyu(degree, -1))


@pytest.mark.parametrize("degree", list(range(3)))
def test_cheb_2nd_call_random_points_multi_input(degree):
    """Test chebyshev polynomial call handles inputs with complex shape"""
    f = smolyay.basis.ChebyshevSecondKind(degree)
    numpy.random.seed(567)
    xs = numpy.random.rand(20, 3, 4, 6) * 2 - 1
    answer = f(xs)
    answer_check = scipy.special.eval_chebyu(degree, xs)
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
    f2.frequency = float(3)
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
    assert f(0) == numpy.exp(0 * c)
    assert f(numpy.pi / 3) == numpy.exp(numpy.pi / 3 * c)
    assert f(3 * numpy.pi / 2) == numpy.exp(3 * numpy.pi / 2 * c)


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
