import importlib
import inspect
import time

import numpy
import pandas

from smolyay.adaptive import make_slow_nested_set
from smolyay.basis import (ChebyshevFirstKind)
from smolyay.grid import (SmolyakGridGenerator)
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *
from compare_surrogates import (compare_error,compare_coefficients,
                                compare_grid_indexes,compare_grid_plot,
                                compare_surrogate_plot)

def print_functions(test_fun_list,names):
    '''Prints functions that will be used in analysis

    Parameters
    ----------
    test_fun_list : list of test_fun objects

    names : names of objects in test_fun_list
    '''
    get_dim = lambda x: x.dim
    d_list = [get_dim(x) for x in test_fun_list]
    counts = numpy.cumsum(pandas.Series(d_list).value_counts(sort=False).values)
    if len(counts) == 1:
        print('Functions with '+str(d_list[0])+' dimensions: ',end='')
        print(*names,sep=' ')
    else:
        counts = numpy.insert(counts,0,0)

        for i in range(len(counts)-1):
            print('Functions with '+str(d_list[counts[i]+1])+
                  ' dimensions: ',end='')
            print(*names[counts[i]:counts[i+1]],sep=' ')

## Initialize parameters
# get test functions
index_names = []
test_functions = []
for name, cls in inspect.getmembers(importlib.import_module("smolyay.test_function_class"), inspect.isclass):
    if not name == 'test_fun':
        test_functions.append(cls())
test_functions.sort(key=lambda x: x.dim) # do faster ones first
index_names = [f.name for f in test_functions]
print('All test functions collected')
# print out functions gathered
print_functions(test_functions,index_names)

## Get Inputs
# functions to test
num_fun_start = len(test_functions)
fun_temp = list(map(str,input("Type names of test functions to analyze " +
                               "(leave blank to choose all): ").split()))
if fun_temp:
    chosen_fun = list(set(fun_temp).intersection(index_names))
    test_fun_list_temp = []
    for ans in chosen_fun:
        fun_index = index_names.index(ans)
        test_fun_list_temp.append(test_functions[fun_index])
    if test_fun_list_temp:
        test_functions = test_fun_list_temp
        test_functions.sort(key=lambda x: x.name)
        test_functions.sort(key=lambda x: x.dim)
        index_names = [f.name for f in test_functions]
    else:
        print('No valid function names.')
# print functions to be used
if len(test_functions) < num_fun_start:
    print('Test functions used: ')
    print_functions(test_functions,index_names)
else:
    print("Analyzing all functions")
# get other inputs
exact = [2,3,4]
try:
    temp = list(map(int,input("Levels of exactness to " +
                               "use (default 2 3 4): ").split()))
    if temp:
        exact = temp
except ValueError:
    pass
points_compare = int(input("Number of points used to " +
                           "check model accuracy (default 5000): ")
                       or "5000")
## Make Grids
grid_norm_list = [SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(exa)) for exa in exact]
grid_slow_list = [SmolyakGridGenerator(make_slow_nested_set(exa)) for exa in exact]
grid_lists = {'Norm' : grid_norm_list,'Slow' : grid_slow_list}
## Do Analysis
a = "Choose the information you want to calculate to do."
b = "(1 = error analysis, 2 = coefficient analysis, 3 = grid index analysis"
c = "4 = 2D plotted grids, 5 = 2D plotted surrogates)"
ana_options = list(input(a + "\n" + b + "\n" + c + "\n").split())
start_time = time.time()
file_header = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(start_time))

if "1" in ana_options:
    compare_error(test_functions,exact,points_compare,grid_norm_list,
                  'Norm',grid_slow_list,'Slow',file_header)
if "2" in ana_options:
    compare_coefficients(test_functions,exact,grid_lists,file_header)
if "3" in ana_options:
    get_dim = lambda x: x.dim
    d_list = numpy.unique([get_dim(x) for x in test_functions])
    compare_grid_indexes(d_list,exact,grid_lists,file_header)

if "4" in ana_options:
    compare_grid_plot(exact,grid_lists,file_header)
if "5" in ana_options:
    plot_funct = []
    for func in test_functions:
        if func.dim == 2:
            plot_funct.append(func)
    compare_surrogate_plot(plot_funct,exact,points_compare,grid_lists,
                           file_header)
print("All requested files created.")
