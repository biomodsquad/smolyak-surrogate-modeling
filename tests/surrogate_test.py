import numpy
import pytest

import smolyay
from smolyay.surrogate import Surrogate

def function_0(x):
    """Test function 0."""
    x1, x2, x3 = x
    return 2*x1 + x2 - x3


def function_0_shifted(x):
    """Test fucntion 0 which is shifted."""
    x1, x2, x3 = x
    return 2*(x1 - 1) + (x2 + 1) - x3


def function_1(x):
    """Test fucntion 1."""
    x1, x2 = x
    return x1 + (2*x2**2 - 1)


def function_2(x):
    """Test funciton 2."""
    return x**2 - 3 * (2 + x) - x


def function_3(x):
    """Test function 3 (gradient)."""
    # function f = x1*x2 - 2*x2
    x1, x2 = x
    return x2, x1 - 2


def function_4(x):
    """Test function 4 (gradient)."""
    # function f = x**3 -2*x
    return 3*x**2 - 2


def branin(x):
    """Branin function."""
    x1, x2 = x
    branin1 = (x2 - 5.1 * x1 ** (2)/(4 * numpy.pi ** 2)
               + 5 * x1 / (numpy.pi) - 6) ** 2
    branin2 = 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(x1)
    branin3 = 10
    branin_function = branin1 + branin2 + branin3
    return branin_function

class Surrogate_Test(Surrogate):
    def predict(self, x):
        return 1
    def fit(self, x, y):
        c = x + y

def test_initialization_1d():
    """Test if class is properly intiallized."""
    grid_generator = smolyay.samples.LatinHypercubeRandomPointSet((-1, 1),50,1234)
    domain = (-1, 1)
    surrogate = Surrogate_Test(grid_generator)
    assert numpy.allclose(surrogate.domain, domain)
    assert isinstance(surrogate.point_set, smolyay.samples.MultidimensionalPointSet)
    assert surrogate.num_dimensions == 1
    assert numpy.allclose(surrogate.points,
                          grid_generator.points)
