import numpy
import importlib
import inspect
import sklearn.metrics

from smolyay.basis import ChebyshevFirstKind
from smolyay.grid import SmolyakGridGenerator, IndexGrid
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *


functions = []
error_data = {}
for name, cls in inspect.getmembers(
        importlib.import_module("smolyay.test_function_class"), 
        inspect.isclass):
    if not name == 'test_fun':
        f = cls()
        functions.append(f)

for fun in functions[0:10]:
    print(fun.name + '  ' + str(fun.dim))
    errors = []
    for exact in range(3,6):
        grid_gen = SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(
            exact))
        surrogate = Surrogate(fun.bounds, grid_gen)
        data = [fun(point) for point in surrogate.points]
        surrogate.train_from_data(data)
        num_points = round(4096**(1/fun.dim))
        test_points = numpy.linspace(fun.lower_bounds,
                fun.upper_bounds,num_points)
        Z_real = numpy.zeros(num_points**fun.dim)
        Z_test = numpy.zeros(num_points**fun.dim)
        for i in range(0,len(Z_real)):
            loc = numpy.unravel_index(i,(num_points,)*fun.dim,order='C')
            coor = list(zip(loc,list(range(len(loc)))))
            x = [test_points[j] for j in coor]
            Z_real[i] = fun(x)
            Z_test[i] = surrogate(x)

        errors.append(sklearn.metrics.max_error(Z_real,Z_test))
    error_data.update({fun.name : errors})

for k in error_data.keys():
    print(k)
    for n in error_data[k]:
        print('%.6g' % (n,),end=', ')
    print('')


