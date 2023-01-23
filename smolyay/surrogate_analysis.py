import numpy
import importlib
import inspect

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator, IndexGrid
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *


functions = []
for name, cls in inspect.getmembers(
        importlib.import_module("smolyay.test_function_class"), 
        inspect.isclass):
    if not name == 'test_fun':
        f = cls()
        functions.append(f)


fun = functions[3]

grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(4))
surrogate = Surrogate([(-5, 10), (0, 15)], grid_gen)
data = [fun(point) for point in surrogate.points]
surrogate.train_from_data(data)
