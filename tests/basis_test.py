import pytest

import numpy
import scipy.special

import smolyay


@pytest.mark.parametrize(
    "cheb_function",
    [
        smolyay.basis.ChebyshevFirstKind,
        smolyay.basis.ChebyshevSecondKind,
    ],
)
def test_cheb_initial(cheb_function):
    """test degrees returns correctly"""
    f2 = cheb_function(2)
    assert f2.degree == 2
    assert isinstance(f2.degree, int)
    assert numpy.array_equal(f2.domain, [-1, 1])
    f2.degree = float(3)
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            1,
            {0.5: 0.5, 1: 1, -1: -1, -0.25: -0.25, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: -0.5, 1: 1, -1: 1, -0.25: -0.875, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 1, 1: 2, -1: -2, -0.25: -0.5, -0.5: -1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 0, 1: 3, -1: 3, -0.25: -0.75, -0.5: 0},
        ),
    ],
)
def test_cheb_call(cheb_function, degree, answer_key):
    """Test chebyshev polynomial call"""
    f = cheb_function(degree)
    assert f(0.5) == pytest.approx(answer_key[0.5])
    assert f(1) == pytest.approx(answer_key[1])
    assert f(-1) == pytest.approx(answer_key[-1])
    assert f(-0.25) == pytest.approx(answer_key[-0.25])
    assert f(-0.5) == pytest.approx(answer_key[-0.5])


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            1,
            {0.5: 0.5, 1: 1, -1: -1, -0.25: -0.25, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: -0.5, 1: 1, -1: 1, -0.25: -0.875, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 1, 1: 2, -1: -2, -0.25: -0.5, -0.5: -1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 0, 1: 3, -1: 3, -0.25: -0.75, -0.5: 0},
        ),
    ],
)
def test_cheb_call_1D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial call with a 1D array"""
    f = cheb_function(degree)
    xs = [0.5, 1, -1, -0.25, -0.5]
    answers = [answer_key[x] for x in xs]
    assert numpy.shape(f(xs)) == numpy.shape(xs)
    assert numpy.allclose(f(xs), answers)

    xs1 = numpy.ones((1, 1)) * xs[0]
    answer1 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            1,
            {0.5: 0.5, 1: 1, -1: -1, -0.25: -0.25, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: -0.5, 1: 1, -1: 1, -0.25: -0.875, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 1, 1: 2, -1: -2, -0.25: -0.5, -0.5: -1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 0, 1: 3, -1: 3, -0.25: -0.75, -0.5: 0},
        ),
    ],
)
def test_cheb_call_2D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial call with a 2D array"""
    f = cheb_function(degree)
    xs = [0.5, 1, -1, -0.25, -0.5, -1, 1, -0.25]
    answers = [answer_key[x] for x in xs]
    xs1 = numpy.reshape(xs, (2, 4))
    answer1 = numpy.reshape(answers, (2, 4))
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 8))
    answer2 = numpy.reshape(answers, (1, 8))
    assert numpy.shape(f(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f(xs2), answer2)

    xs3 = numpy.reshape(xs, (8, 1))
    answer3 = numpy.reshape(answers, (8, 1))
    assert numpy.shape(f(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f(xs3), answer3)

    xs4 = numpy.ones((1, 1)) * xs[0]
    answer4 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f(xs4), answer4)


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            1,
            {0.5: 0.5, 1: 1, -1: -1, -0.25: -0.25, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: -0.5, 1: 1, -1: 1, -0.25: -0.875, -0.5: -0.5},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 1, 1: 2, -1: -2, -0.25: -0.5, -0.5: -1},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 0, 1: 3, -1: 3, -0.25: -0.75, -0.5: 0},
        ),
    ],
)
def test_cheb_call_3D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial call with a 3D array"""
    f = cheb_function(degree)
    xs = [
        0.5,
        1,
        -1,
        -0.25,
        -0.5,
        -1,
        1,
        -0.25,
        -0.25,
        1,
        -1,
        0.5,
        0.5,
        1,
        1,
        -0.25,
        -0.5,
        -1,
        1,
        -0.25,
        -0.25,
        1,
        1,
        0.5,
    ]
    answers = [answer_key[x] for x in xs]

    xs1 = numpy.reshape(xs, (2, 3, 4))
    answer1 = numpy.reshape(answers, (2, 3, 4))
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 1, 24))
    answer2 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f(xs2), answer2)

    xs3 = numpy.reshape(xs, (1, 24, 1))
    answer3 = numpy.reshape(answers, (1, 24, 1))
    assert numpy.shape(f(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f(xs3), answer3)

    xs4 = numpy.reshape(xs, (1, 1, 24))
    answer4 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f(xs4), answer4)

    xs5 = numpy.reshape(xs, (1, 6, 4))
    answer5 = numpy.reshape(answers, (1, 6, 4))
    assert numpy.shape(f(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f(xs5), answer5)

    xs6 = numpy.reshape(xs, (6, 4, 1))
    answer6 = numpy.reshape(answers, (6, 4, 1))
    assert numpy.shape(f(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f(xs6), answer6)

    xs7 = numpy.reshape(xs, (6, 1, 4))
    answer7 = numpy.reshape(answers, (6, 1, 4))
    assert numpy.shape(f(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f(xs7), answer7)

    xs8 = numpy.ones((1, 1, 1)) * xs[0]
    answer8 = numpy.ones((1, 1, 1)) * answers[0]
    assert numpy.shape(f(xs8)) == numpy.shape(xs8)
    assert numpy.allclose(f(xs8), answer8)


@pytest.mark.parametrize(
    "cheb_function",
    [
        smolyay.basis.ChebyshevFirstKind,
        smolyay.basis.ChebyshevSecondKind,
    ],
)
def test_cheb_call_invalid_input(cheb_function):
    """Test call raises error if input is outside domain [-1, 1]"""
    f = cheb_function(4)
    with pytest.raises(ValueError):
        f(2)
    with pytest.raises(ValueError):
        f(-2)
    with pytest.raises(ValueError):
        f([0.5, 0.7, 3, 0.8])
    with pytest.raises(ValueError):
        f([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0}),
        (smolyay.basis.ChebyshevFirstKind, 1, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: 2, 1: 4, -1: -4, -0.25: -1, -0.5: -2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 2, 1: 2, -1: 2, -0.25: 2, -0.5: 2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 4, 1: 8, -1: -8, -0.25: -2, -0.5: -4},
        ),
    ],
)
def test_cheb_derivative(cheb_function, degree, answer_key):
    """Test chebyshev polynomial derivative"""
    f = cheb_function(degree)
    assert f.derivative(0.5) == pytest.approx(answer_key[0.5])
    assert f.derivative(1) == pytest.approx(answer_key[1])
    assert f.derivative(-1) == pytest.approx(answer_key[-1])
    assert f.derivative(-0.25) == pytest.approx(answer_key[-0.25])
    assert f.derivative(-0.5) == pytest.approx(answer_key[-0.5])


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0}),
        (smolyay.basis.ChebyshevFirstKind, 1, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: 2, 1: 4, -1: -4, -0.25: -1, -0.5: -2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 2, 1: 2, -1: 2, -0.25: 2, -0.5: 2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 4, 1: 8, -1: -8, -0.25: -2, -0.5: -4},
        ),
    ],
)
def test_cheb_derivative_1D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial derivative with a 1D array"""
    f = cheb_function(degree)
    xs = [0.5, 1, -1, -0.25, -0.5]
    answers = [answer_key[x] for x in xs]
    assert numpy.shape(f.derivative(xs)) == numpy.shape(xs)
    assert numpy.allclose(f.derivative(xs), answers)

    xs1 = numpy.ones((1, 1)) * xs[0]
    answer1 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0}),
        (smolyay.basis.ChebyshevFirstKind, 1, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: 2, 1: 4, -1: -4, -0.25: -1, -0.5: -2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 2, 1: 2, -1: 2, -0.25: 2, -0.5: 2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 4, 1: 8, -1: -8, -0.25: -2, -0.5: -4},
        ),
    ],
)
def test_cheb_derivative_2D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial derivative with a 2D array"""
    f = cheb_function(degree)
    xs = [0.5, 1, -1, -0.25, -0.5, -1, 1, -0.25]
    answers = [answer_key[x] for x in xs]
    xs1 = numpy.reshape(xs, (2, 4))
    answer1 = numpy.reshape(answers, (2, 4))
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 8))
    answer2 = numpy.reshape(answers, (1, 8))
    assert numpy.shape(f.derivative(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f.derivative(xs2), answer2)

    xs3 = numpy.reshape(xs, (8, 1))
    answer3 = numpy.reshape(answers, (8, 1))
    assert numpy.shape(f.derivative(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f.derivative(xs3), answer3)

    xs4 = numpy.ones((1, 1)) * xs[0]
    answer4 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f.derivative(xs4), answer4)


@pytest.mark.parametrize(
    "cheb_function,degree,answer_key",
    [
        (smolyay.basis.ChebyshevFirstKind, 0, {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0}),
        (smolyay.basis.ChebyshevFirstKind, 1, {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1}),
        (
            smolyay.basis.ChebyshevFirstKind,
            2,
            {0.5: 2, 1: 4, -1: -4, -0.25: -1, -0.5: -2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            0,
            {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            1,
            {0.5: 2, 1: 2, -1: 2, -0.25: 2, -0.5: 2},
        ),
        (
            smolyay.basis.ChebyshevSecondKind,
            2,
            {0.5: 4, 1: 8, -1: -8, -0.25: -2, -0.5: -4},
        ),
    ],
)
def test_cheb_derivative_3D(cheb_function, degree, answer_key):
    """Test chebyshev polynomial derivative with a 3D array"""
    f = cheb_function(degree)
    xs = [
        0.5,
        1,
        -1,
        -0.25,
        -0.5,
        -1,
        1,
        -0.25,
        -0.25,
        1,
        -1,
        0.5,
        0.5,
        1,
        1,
        -0.25,
        -0.5,
        -1,
        1,
        -0.25,
        -0.25,
        1,
        1,
        0.5,
    ]
    answers = [answer_key[x] for x in xs]

    xs1 = numpy.reshape(xs, (2, 3, 4))
    answer1 = numpy.reshape(answers, (2, 3, 4))
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 1, 24))
    answer2 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f.derivative(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f.derivative(xs2), answer2)

    xs3 = numpy.reshape(xs, (1, 24, 1))
    answer3 = numpy.reshape(answers, (1, 24, 1))
    assert numpy.shape(f.derivative(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f.derivative(xs3), answer3)

    xs4 = numpy.reshape(xs, (1, 1, 24))
    answer4 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f.derivative(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f.derivative(xs4), answer4)

    xs5 = numpy.reshape(xs, (1, 6, 4))
    answer5 = numpy.reshape(answers, (1, 6, 4))
    assert numpy.shape(f.derivative(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f.derivative(xs5), answer5)

    xs6 = numpy.reshape(xs, (6, 4, 1))
    answer6 = numpy.reshape(answers, (6, 4, 1))
    assert numpy.shape(f.derivative(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f.derivative(xs6), answer6)

    xs7 = numpy.reshape(xs, (6, 1, 4))
    answer7 = numpy.reshape(answers, (6, 1, 4))
    assert numpy.shape(f.derivative(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f.derivative(xs7), answer7)

    xs8 = numpy.ones((1, 1, 1)) * xs[0]
    answer8 = numpy.ones((1, 1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs8)) == numpy.shape(xs8)
    assert numpy.allclose(f.derivative(xs8), answer8)


@pytest.mark.parametrize(
    "cheb_function",
    [
        smolyay.basis.ChebyshevFirstKind,
        smolyay.basis.ChebyshevSecondKind,
    ],
)
def test_cheb_derivative_invalid_input(cheb_function):
    """Test call raises error if input is outside domain [-1, 1]"""
    f = cheb_function(4)
    with pytest.raises(ValueError):
        f.derivative(2)
    with pytest.raises(ValueError):
        f.derivative(-2)
    with pytest.raises(ValueError):
        f.derivative([[0.5, 0.7, 3, 0.8], [0.5, 0.7, 0.8, -0.5]])


def test_cheb_call_extrema_points():
    """Test chebyshev polynomial at extrema"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    extrema_points = [-1.0, -1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2), 1]
    extrema_output = [1, -1, 1, -1, 1]
    assert numpy.allclose(f(extrema_points), extrema_output)


def test_cheb_call_root_points():
    """Test chebyshev polynomial at roots"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    root_points = [
        -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
    ]
    assert numpy.allclose(f(root_points), numpy.zeros(4))


def test_cheb_2nd_call_root_points():
    """Test chebyshev polynomial roots"""
    f = smolyay.basis.ChebyshevSecondKind(3)
    root_points = [-1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2)]
    assert numpy.allclose(f(root_points), numpy.zeros(3))


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
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 1,
                numpy.pi / 3: 1,
                3 * numpy.pi / 2: 1,
                -0.25: 1,
                -0.5: 1,
                numpy.pi / 6: 1,
                2 * numpy.pi: 1,
            },
        ),
        (
            1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_call(frequency, answer_key):
    """Test Trigonometric function call"""
    f = smolyay.basis.Trigonometric(frequency)
    assert f(0) == pytest.approx(answer_key[0])
    assert f(numpy.pi / 3) == pytest.approx(answer_key[numpy.pi / 3])
    assert f(3 * numpy.pi / 2) == pytest.approx(answer_key[3 * numpy.pi / 2])
    assert f(numpy.pi / 6) == pytest.approx(answer_key[numpy.pi / 6])
    assert f(2 * numpy.pi) == pytest.approx(answer_key[2 * numpy.pi])


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 1,
                numpy.pi / 3: 1,
                3 * numpy.pi / 2: 1,
                -0.25: 1,
                -0.5: 1,
                numpy.pi / 6: 1,
                2 * numpy.pi: 1,
            },
        ),
        (
            1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_call_1D(frequency, answer_key):
    """Test Trigonometric function call with a 1D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [0, numpy.pi / 3, 3 * numpy.pi / 2, numpy.pi / 6, 2 * numpy.pi]
    answers = [answer_key[x] for x in xs]
    assert numpy.shape(f(xs)) == numpy.shape(xs)
    assert numpy.allclose(f(xs), answers)

    xs1 = numpy.ones((1, 1)) * xs[0]
    answer1 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 1,
                numpy.pi / 3: 1,
                3 * numpy.pi / 2: 1,
                -0.25: 1,
                -0.5: 1,
                numpy.pi / 6: 1,
                2 * numpy.pi: 1,
            },
        ),
        (
            1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_call_2D(frequency, answer_key):
    """Test Trigonometric function call with a 2D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        numpy.pi / 6,
        3 * numpy.pi / 2,
        2 * numpy.pi,
    ]
    answers = [answer_key[x] for x in xs]
    xs1 = numpy.reshape(xs, (2, 4))
    answer1 = numpy.reshape(answers, (2, 4))
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 8))
    answer2 = numpy.reshape(answers, (1, 8))
    assert numpy.shape(f(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f(xs2), answer2)

    xs3 = numpy.reshape(xs, (8, 1))
    answer3 = numpy.reshape(answers, (8, 1))
    assert numpy.shape(f(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f(xs3), answer3)

    xs4 = numpy.ones((1, 1)) * xs[0]
    answer4 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f(xs4), answer4)


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 1,
                numpy.pi / 3: 1,
                3 * numpy.pi / 2: 1,
                -0.25: 1,
                -0.5: 1,
                numpy.pi / 6: 1,
                2 * numpy.pi: 1,
            },
        ),
        (
            1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: 1,
                numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_call_3D(frequency, answer_key):
    """Test Trigonometric function call with a 3D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [
        numpy.pi / 3,
        3 * numpy.pi / 2,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        3 * numpy.pi / 2,
        numpy.pi / 3,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        3 * numpy.pi / 2,
        numpy.pi / 6,
        0,
        numpy.pi / 6,
        3 * numpy.pi / 2,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
    ]
    answers = [answer_key[x] for x in xs]

    xs1 = numpy.reshape(xs, (2, 3, 4))
    answer1 = numpy.reshape(answers, (2, 3, 4))
    assert numpy.shape(f(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 1, 24))
    answer2 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f(xs2), answer2)

    xs3 = numpy.reshape(xs, (1, 24, 1))
    answer3 = numpy.reshape(answers, (1, 24, 1))
    assert numpy.shape(f(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f(xs3), answer3)

    xs4 = numpy.reshape(xs, (1, 1, 24))
    answer4 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f(xs4), answer4)

    xs5 = numpy.reshape(xs, (1, 6, 4))
    answer5 = numpy.reshape(answers, (1, 6, 4))
    assert numpy.shape(f(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f(xs5), answer5)

    xs6 = numpy.reshape(xs, (6, 4, 1))
    answer6 = numpy.reshape(answers, (6, 4, 1))
    assert numpy.shape(f(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f(xs6), answer6)

    xs7 = numpy.reshape(xs, (6, 1, 4))
    answer7 = numpy.reshape(answers, (6, 1, 4))
    assert numpy.shape(f(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f(xs7), answer7)

    xs8 = numpy.ones((1, 1, 1)) * xs[0]
    answer8 = numpy.ones((1, 1, 1)) * answers[0]
    assert numpy.shape(f(xs8)) == numpy.shape(xs8)
    assert numpy.allclose(f(xs8), answer8)


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 0,
                numpy.pi / 3: 0,
                3 * numpy.pi / 2: 0,
                -0.25: 0,
                -0.5: 0,
                numpy.pi / 6: 0,
                2 * numpy.pi: 0,
            },
        ),
        (
            1,
            {
                0: 1j,
                numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: -1j,
                numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_derivative(frequency, answer_key):
    """Test Trigonometric function derivative"""
    f = smolyay.basis.Trigonometric(frequency)
    assert f.derivative(0) == pytest.approx(answer_key[0])
    assert f.derivative(numpy.pi / 3) == pytest.approx(answer_key[numpy.pi / 3])
    assert f.derivative(3 * numpy.pi / 2) == pytest.approx(answer_key[3 * numpy.pi / 2])
    assert f.derivative(numpy.pi / 6) == pytest.approx(answer_key[numpy.pi / 6])
    assert f.derivative(2 * numpy.pi) == pytest.approx(answer_key[2 * numpy.pi])


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 0,
                numpy.pi / 3: 0,
                3 * numpy.pi / 2: 0,
                -0.25: 0,
                -0.5: 0,
                numpy.pi / 6: 0,
                2 * numpy.pi: 0,
            },
        ),
        (
            1,
            {
                0: 1j,
                numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: -1j,
                numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_derivative_1D(frequency, answer_key):
    """Test Trigonometric function derivative with a 1D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [0, numpy.pi / 3, 3 * numpy.pi / 2, numpy.pi / 6, 2 * numpy.pi]
    answers = [answer_key[x] for x in xs]
    assert numpy.shape(f.derivative(xs)) == numpy.shape(xs)
    assert numpy.allclose(f.derivative(xs), answers)

    xs1 = numpy.ones((1, 1)) * xs[0]
    answer1 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 0,
                numpy.pi / 3: 0,
                3 * numpy.pi / 2: 0,
                -0.25: 0,
                -0.5: 0,
                numpy.pi / 6: 0,
                2 * numpy.pi: 0,
            },
        ),
        (
            1,
            {
                0: 1j,
                numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: -1j,
                numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_derivative_2D(frequency, answer_key):
    """Test Trigonometric function derivative  with a 2D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        numpy.pi / 6,
        3 * numpy.pi / 2,
        2 * numpy.pi,
    ]
    answers = [answer_key[x] for x in xs]
    xs1 = numpy.reshape(xs, (2, 4))
    answer1 = numpy.reshape(answers, (2, 4))
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 8))
    answer2 = numpy.reshape(answers, (1, 8))
    assert numpy.shape(f.derivative(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f.derivative(xs2), answer2)

    xs3 = numpy.reshape(xs, (8, 1))
    answer3 = numpy.reshape(answers, (8, 1))
    assert numpy.shape(f.derivative(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f.derivative(xs3), answer3)

    xs4 = numpy.ones((1, 1)) * xs[0]
    answer4 = numpy.ones((1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f.derivative(xs4), answer4)


@pytest.mark.parametrize(
    "frequency,answer_key",
    [
        (
            0,
            {
                0: 0,
                numpy.pi / 3: 0,
                3 * numpy.pi / 2: 0,
                -0.25: 0,
                -0.5: 0,
                numpy.pi / 6: 0,
                2 * numpy.pi: 0,
            },
        ),
        (
            1,
            {
                0: 1j,
                numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j),
                3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j),
                2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j),
            },
        ),
        (
            -1,
            {
                0: -1j,
                numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j * -1),
                3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j * -1),
                2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j * -1),
            },
        ),
    ],
)
def test_trig_derivative_3D(frequency, answer_key):
    """Test Trigonometric function derivative with a 3D array"""
    f = smolyay.basis.Trigonometric(frequency)
    xs = [
        numpy.pi / 3,
        3 * numpy.pi / 2,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        3 * numpy.pi / 2,
        numpy.pi / 3,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        3 * numpy.pi / 2,
        numpy.pi / 6,
        0,
        numpy.pi / 6,
        3 * numpy.pi / 2,
        2 * numpy.pi,
        0,
        numpy.pi / 3,
        numpy.pi / 6,
        2 * numpy.pi,
    ]
    answers = [answer_key[x] for x in xs]

    xs1 = numpy.reshape(xs, (2, 3, 4))
    answer1 = numpy.reshape(answers, (2, 3, 4))
    assert numpy.shape(f.derivative(xs1)) == numpy.shape(xs1)
    assert numpy.allclose(f.derivative(xs1), answer1)

    xs2 = numpy.reshape(xs, (1, 1, 24))
    answer2 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f.derivative(xs2)) == numpy.shape(xs2)
    assert numpy.allclose(f.derivative(xs2), answer2)

    xs3 = numpy.reshape(xs, (1, 24, 1))
    answer3 = numpy.reshape(answers, (1, 24, 1))
    assert numpy.shape(f.derivative(xs3)) == numpy.shape(xs3)
    assert numpy.allclose(f.derivative(xs3), answer3)

    xs4 = numpy.reshape(xs, (1, 1, 24))
    answer4 = numpy.reshape(answers, (1, 1, 24))
    assert numpy.shape(f.derivative(xs4)) == numpy.shape(xs4)
    assert numpy.allclose(f.derivative(xs4), answer4)

    xs5 = numpy.reshape(xs, (1, 6, 4))
    answer5 = numpy.reshape(answers, (1, 6, 4))
    assert numpy.shape(f.derivative(xs5)) == numpy.shape(xs5)
    assert numpy.allclose(f.derivative(xs5), answer5)

    xs6 = numpy.reshape(xs, (6, 4, 1))
    answer6 = numpy.reshape(answers, (6, 4, 1))
    assert numpy.shape(f.derivative(xs6)) == numpy.shape(xs6)
    assert numpy.allclose(f.derivative(xs6), answer6)

    xs7 = numpy.reshape(xs, (6, 1, 4))
    answer7 = numpy.reshape(answers, (6, 1, 4))
    assert numpy.shape(f.derivative(xs7)) == numpy.shape(xs7)
    assert numpy.allclose(f.derivative(xs7), answer7)

    xs8 = numpy.ones((1, 1, 1)) * xs[0]
    answer8 = numpy.ones((1, 1, 1)) * answers[0]
    assert numpy.shape(f.derivative(xs8)) == numpy.shape(xs8)
    assert numpy.allclose(f.derivative(xs8), answer8)


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
