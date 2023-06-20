import pytest
import warnings

import numpy
import importlib
import inspect

from smolyay.basis import ChebyshevFirstKind
import smolyay.benchmark
from smolyay.benchmark import (BenchmarkFunction, branin)
from smolyay.grid import SmolyakGridGenerator
from smolyay.surrogate import Surrogate


class TestClass1D(BenchmarkFunction):
    """A test class for testing BenchmarkFunction objects with 1 dimension"""
    @property
    def domain(self):
        return [[0.4,10]]
    def _function(self,x):
        return numpy.squeeze(4*0.5*((2/x)**12 - (2/x)**6))

class TestClass3D(BenchmarkFunction):
    """A test class for testing BenchmarkFunction objects with >1 dimensions"""
    @property
    def domain(self):
        return [[0,10],[-5,-1],[11,13]]
    def _function(self,x):
        return x[...,0]**0.5 + x[...,1]*3 - x[...,2]/3 + x[...,0]/x[...,1]

"""List of non-abstract class objects used as benchmark functions"""
functions = []
for name, cls in inspect.getmembers(
        importlib.import_module("smolyay.benchmark"), inspect.isclass):
    if not inspect.isabstract(cls):
        f = cls()
        functions.append(f)

@pytest.fixture
def test_class_1():
    return TestClass1D()

@pytest.fixture
def test_class_3():
    return TestClass3D()

def test_name(test_class_1,test_class_3):
    """Test name is the name of the function"""
    assert test_class_1.name == "TestClass1D"
    assert test_class_3.name == "TestClass3D"

def test_domain(test_class_1,test_class_3):
    """Test return domain as list of lists"""
    assert test_class_1.domain == [[0.4,10]]
    assert test_class_3.domain == [[0,10],[-5,-1],[11,13]]

def test_lower_bounds(test_class_1,test_class_3):
    """Test return lower bounds"""
    assert test_class_1.lower_bounds == [0.4]
    assert test_class_3.lower_bounds == [0,-5,11]

def test_upper_bounds(test_class_1,test_class_3):
    """Test return upper bounds"""
    assert test_class_1.upper_bounds == [10]
    assert test_class_3.upper_bounds == [10,-1,13]

def test_dimension(test_class_1,test_class_3):
    """Test return dimension"""
    assert test_class_1.dimension == 1
    assert test_class_3.dimension == 3

@pytest.mark.filterwarnings("error")
def test_call_no_error_1(test_class_1,test_class_3):
    """Test valid inputs of call give no errors for 1D functions"""
    x_1d = numpy.linspace(test_class_1.lower_bounds,test_class_1.upper_bounds)
    for x in x_1d:
        test_class_1(x)

@pytest.mark.filterwarnings("error")
def test_call_no_error_3D(test_class_3):
    """Test valid inputs of call give no errors for >1D functions"""
    x_3d = numpy.linspace(test_class_3.lower_bounds,test_class_3.upper_bounds)
    for x in x_3d:
        test_class_3(x)


def test_call_error_1D(test_class_1):
    """Test if there is an error for inputs out of bounds for 1D functions"""
    with pytest.raises(ValueError):
        test_class_1(0)
    with pytest.raises(ValueError):
        test_class_1(20)

def test_call_error_3D(test_class_3):
    """Test if there is an error for inputs out of bounds for >1D functions"""
    with pytest.raises(ValueError):
        test_class_3([-2,-2,12])
    with pytest.raises(ValueError):
        test_class_3([11,-2,12])
    with pytest.raises(ValueError):
        test_class_3([5,-10,12])
    with pytest.raises(ValueError):
        test_class_3([5,4,12])
    with pytest.raises(ValueError):
        test_class_3([5,-2,6])
    with pytest.raises(ValueError):
        test_class_3([5,-2,24])

@pytest.mark.filterwarnings("error")
def test_call_no_error_multi_input_1D(test_class_1):
    """Test multiple inputs of call give no errors for 1D functions"""
    x_1d = numpy.linspace(test_class_1.lower_bounds,test_class_1.upper_bounds)
    y_1d_multi = test_class_1(x_1d)
    y_1d = [test_class_1(x) for x in x_1d]
    y_1d_fun = [test_class_1._function(x) for x in x_1d]
    y_1d_multi_fun = test_class_1._function(x_1d)
    assert len(y_1d_multi) == len(x_1d)
    assert numpy.array_equiv(y_1d,y_1d_multi)
    assert numpy.array_equiv(y_1d_fun,y_1d_multi_fun)
    assert numpy.array_equiv(y_1d_multi,y_1d_multi_fun)

@pytest.mark.filterwarnings("error")
def test_call_no_error_multi_input_3D(test_class_3):
    """Test multiple inputs of call give no errors for >1D functions"""
    x_3d = numpy.linspace(test_class_3.lower_bounds,test_class_3.upper_bounds)
    y_3d_multi = test_class_3(x_3d)
    y_3d = [test_class_3(x) for x in x_3d]
    y_3d_fun = numpy.squeeze([test_class_3._function(x) for x in x_3d])
    y_3d_multi_fun = test_class_3._function(x_3d)
    assert len(y_3d_multi) == len(x_3d)
    assert numpy.array_equiv(y_3d,y_3d_multi)
    assert numpy.array_equiv(y_3d_fun,y_3d_multi_fun)
    assert numpy.array_equiv(y_3d_multi,y_3d_multi_fun)



def test_call_error_1D_multi_input(test_class_1):
    """Test if there is an error for inputs out of bounds for 1D functions"""
    with pytest.raises(ValueError):
        test_class_1([0.5, 0.6, 1.7, 0, 9.2])
    with pytest.raises(ValueError):
        test_class_1([20, 5, 6, 2])

def test_call_error_3D_multi_input(test_class_3):
    """Test if there is an error for inputs out of bounds for >1D functions"""
    with pytest.raises(ValueError):
        test_class_3([[-2,-2,12],[3,-1,11.5],[4,-4,11.1],[1,-4.4,12.7]])
    with pytest.raises(ValueError):
        test_class_3([[1,-1.1,11.3],[9,-1.2,11.4],[11,-2,12]])
    with pytest.raises(ValueError):
        test_class_3([[1,-1,11.5],[5,-10,12],[2,-2,13],[3,-3,12]])
    with pytest.raises(ValueError):
        test_class_3([[5,4,12],[3,6,12]])
    with pytest.raises(ValueError):
        test_class_3([[0,-1,11],[5,-2,6]])
    with pytest.raises(ValueError):
        test_class_3([[5,-2,24],[1,-2,11]])

@pytest.mark.parametrize("fun",functions)
def test_functions_domain_match_dimension(fun):
    """Test that all domains are the correct shape"""
    assert len(fun.domain) == fun.dimension
    for bound in fun.domain:
        assert len(bound) == 2

@pytest.mark.parametrize("fun",functions)
def test_functions_good_bounds(fun):
    """Test if all lower bounds are lower than upper bounds"""
    for bound in fun.domain:
        assert bound[0] < bound[1]

@pytest.mark.parametrize("fun",functions)
@pytest.mark.filterwarnings("error")
def test_functions_call(fun):
    """Test all functions do not raise error or warning for inputs in bounds"""
    x_list = numpy.linspace(fun.lower_bounds,fun.upper_bounds)
    y_call = [fun(i) for i in x_list]
    y_call_multi = fun(x_list)
    y_fun = numpy.squeeze([fun._function(i) for i in x_list])
    y_fun_multi = fun._function(x_list)
    assert numpy.array_equiv(y_call,y_call_multi)
    assert numpy.array_equiv(y_fun,y_fun_multi)
    assert numpy.array_equiv(y_call_multi,y_fun_multi)


@pytest.mark.parametrize("fun",functions)
def test_functions_call_error_lower(fun):
    """Test all functions error for inputs below bounds"""
    below_bounds = numpy.add(fun.lower_bounds, -5)
    with pytest.raises(ValueError):
        fun(below_bounds)

@pytest.mark.parametrize("fun",functions)
def test_functions_call_error_above(fun):
    """Test all functions error for inputs above bounds"""
    above_bounds = numpy.add(fun.upper_bounds,5)
    with pytest.raises(ValueError):
        fun(above_bounds)

def test_surrogate():
    """Test use of function class with surrogate"""
    a = branin()
    grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4))
    surrogate = Surrogate(a.domain, grid_gen)
    data = [a(point) for point in surrogate.points]
    surrogate.train_from_data(data)
    assert numpy.isclose(surrogate([6,0.5]),a([6,0.5]))

