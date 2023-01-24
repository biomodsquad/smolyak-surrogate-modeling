import pytest

import numpy
import importlib
import inspect

from smolyay.test_function_class import *
from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator
from smolyay.surrogate import Surrogate


functions = []
for name, cls in inspect.getmembers(
        importlib.import_module("smolyay.test_function_class"), 
        inspect.isclass):
    if not name == 'test_fun':
        f = cls()
        functions.append(f)


def test_call():
    """Test all functions do not raise error for inputs in bounds"""
    for f in functions:
        inputs = numpy.linspace(f.lower_bounds,f.upper_bounds)
        for i in inputs:
            f(i)

def test_call_error_lower():
    """Test all functions error for inputs below bounds"""
    for f in functions:
        below_bounds = numpy.add(f.lower_bounds, -5)
        with pytest.raises(ValueError):
            f(below_bounds)

def test_call_error_above():
    """Test all functions error for inputs above bounds"""
    for f in functions:
        above_bounds = numpy.add(f.upper_bounds,5)
        with pytest.raises(ValueError):
            f(above_bounds)

def test_bounds():
    """Test return bounds as list of tuples"""
    a = branin()
    assert a.bounds == [(-5,10),(0,15)]

def test_surrogate():
    """Test use of function class with surrogate"""
    a = branin()
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4))
    surrogate = Surrogate(a.bounds, grid_gen)
    data = [a(point) for point in surrogate.points]
    surrogate.train_from_data(data)
    assert numpy.isclose(surrogate([6,0.5]),a([6,0.5]))

