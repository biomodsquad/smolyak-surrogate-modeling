import importlib
import inspect
import itertools
import time

import matplotlib
import numpy
import pandas

from smolyay.adaptive import make_slow_nested_set
from smolyay.basis import (ChebyshevFirstKind, BasisFunctionSet,
                           NestedBasisFunctionSet)
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator, generate_compositions)
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *

print('Running...')
exact = [2,3,4]
points_along_each_dim = 50
grid_norm_list = [SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(exa)) for exa in exact]
grid_slow_list = [SmolyakGridGenerator(make_slow_nested_set(exa)) for exa in exact]

index_names = []
test_functions = []

for name, cls in inspect.getmembers(importlib.import_module("test_function_class"), inspect.isclass):
    if not name == 'test_fun':
        test_functions.append(cls())
test_functions.sort(key=lambda x: x.dim)
index_names = [f.name for f in test_functions]
print('All test functions collected')
error_data_collection = numpy.zeros((len(test_functions),len(exact)*3))
print('Begin calculations.')
start_time = time.time()
try:
    for (func,j) in zip(test_functions,range(len(test_functions))):
        func_time_start = time.time()
        # create test points
        print('Estimating function : ' + func.name)
        test_points = numpy.linspace(func.lower_bounds,
                                  func.upper_bounds,points_along_each_dim)
        for (grid_norm,grid_slow,k) in zip(grid_norm_list,grid_slow_list,range(len(exact))):
            calc_time = time.time()
            surrogate_norm = Surrogate(func.bounds,grid_norm)
            surrogate_slow = Surrogate(func.bounds,grid_slow)
            data_norm = [func(point) for point in surrogate_norm.points]
            data_slow = [func(point) for point in surrogate_slow.points]
            surrogate_norm.train_from_data(data_norm)
            surrogate_slow.train_from_data(data_slow)
            calc_time_end = time.time() - calc_time
            # test the error of surrogates
            error_norm = 0
            error_slow = 0
            for i in range(points_along_each_dim**func.dim):
                loc = numpy.unravel_index(i,(points_along_each_dim,)*func.dim,order='F')
                coor = list(zip(loc,list(range(len(loc)))))
                x = [test_points[m] for m in coor]
                error_norm += (func(x) - surrogate_norm(x))**2
                error_slow += (func(x) - surrogate_slow(x))**2
            error_data_collection[j,k] = error_norm
            error_data_collection[j,k+len(exact)] = error_slow
            error_data_collection[j,k+2*len(exact)] = calc_time_end
        print('  Time for '+func.name+': ' +
              str(round(time.time()-func_time_start,2))+' seconds')
finally:
    print('All done.')
    end_time = time.time()
    total_time = end_time-start_time # time to finish executing
    print('Time to calculate: ' + str(total_time//60) +
          ' min ' + str(total_time % 60) + ' sec')
    print('Time to calculate: ' +  str(total_time) + ' sec')
    print('')
    head_names = ['Norm µ=','Slow µ=','Runtime µ=']
    column_names = [''.join(x) for x in list(itertools.product(head_names,map(str,exact)))]
    results = pandas.DataFrame(error_data_collection,index=index_names,columns=column_names)
    date_t = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(start_time))
    file_name = date_t + 'slow_results.csv'
    results.to_csv(file_name)




        
