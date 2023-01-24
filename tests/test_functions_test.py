import pytest

import numpy
import importlib
import inspect

from smolyay.test_function_class import *



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
