import argparse
import importlib
import inspect
import os
import time

import numpy
import pandas

from smolyay.adaptive import make_slow_nested_set
from smolyay.basis import (ChebyshevFirstKind, BasisFunctionSet, 
                           NestedBasisFunctionSet)
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator)
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *
from compare_surrogates import (compare_error,compare_coefficients,
                                compare_grid_indexes,compare_grid_plot,
                                compare_surrogate_plot)

## Grab arguments
# get defaults
start_time = time.time()
get_dim = lambda x : x.dim
index_names = []
test_functions = []
for name, cls in inspect.getmembers(importlib.import_module("smolyay.test_function_class"), inspect.isclass):
    if not name == 'test_fun':
        test_functions.append(cls())
test_functions.sort(key=lambda x: x.dim) # do faster ones first
function_names = [f.name for f in test_functions]
d_list = numpy.unique([str(get_dim(x)) for x in test_functions])
# get arguments
parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("--analysis_options",required=True,nargs="+",
                    help=("surrogate function properties to analyze. Valid "+
                          "options are \'error\', \'coeff\', \'indexes\', " +
                          "\'gridplot\',\'surrogateplot\'"))

parser.add_argument("--exactness_list",required=True,nargs="+",
                    default=[2,3,4],
                    help=("list of exactnesses that corresponds to the "+
                          "grids in grid_lists"))
parser.add_argument("--grid_lists",required=False,
                    help=("a dictionary of lists of IndexGridGenerator "+
                          "objects"))
# optional arguments
parser.add_argument("--function_and_dimension",required=False,nargs="+",
                    default=function_names,help="list of functions to test, "+
                    "given by name or dimension")
parser.add_argument("--points_compare",required=False,nargs="?",default=5000,
                    help="number of points used for compare error")
parser.add_argument("--points_plot",required=False,nargs="?",default=50,
                    help="number of points used for 2D surrogate plots")
parser.add_argument("--seed",required=False,nargs="?",default=start_time,
                    help="the random seed for compare error")

args = parser.parse_args()

## Initialize parameters
# Required for option error: test_functions,points_compare,seed
# Required for option coeff: test_functions 
# Required for option indexes: test_functions (only needs dimensions)
# Required for option gridplot: 
# Required for option surrogateplot: test_functions,points_plot
exactness_list = [int(x) for x in args.exactness_list if x.isdigit()]
points_compare = int(args.points_compare)
points_plot = int(args.points_plot)
seed = int(args.seed)


## Get test functions and dimensions
# get test functions available
chosen_by_name = list(set(args.function_and_dimension).intersection(function_names))
chosen_by_dim = list(set(args.function_and_dimension).intersection(d_list))
chosen_fun_set = set()
chosen_dim_set = set([int(x) for x in args.function_and_dimension if x.isdigit()])
# add functions given by name and dimension
for f in test_functions:
    if (str(f.dim) in chosen_by_dim) or (f.name in chosen_by_name):
        chosen_fun_set.add(f)
        chosen_dim_set.add(f.dim)
# sort functions
if chosen_fun_set:
    test_functions = list(chosen_fun_set)
    test_functions.sort(key=lambda x: x.name)
    test_functions.sort(key=lambda x: x.dim)
    function_names = [f.name for f in test_functions]
    test_dimensions = list(chosen_dim_set)
    test_dimensions.sort()

## Make Grids
# turn this part into bash somehow
##grid_norm_list = [SmolyakGridGenerator(ChebyshevFirstKind.make_nested_set(exa))
##                  for exa in args.exactness_list]
##grid_slow_list = [SmolyakGridGenerator(make_slow_nested_set(exa))
##                  for exa in args.exactness_list]
grid_norm_list = [TensorGridGenerator(ChebyshevFirstKind.make_nested_set(exa))
                  for exa in exactness_list]
grid_slow_list = [TensorGridGenerator(make_slow_nested_set(exa))
                  for exa in exactness_list]
grid_lists = {'Norm' : grid_norm_list,'Slow' : grid_slow_list}
# create file header
time_header = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(start_time))
folder_name = './analysisopt_'+ time_header
folder = os.makedirs(folder_name)
file_header = folder_name + '/'
## Do Analysis
if "error" in args.analysis_options:
    compare_error(test_functions,exactness_list,args.points_compare,
                  grid_lists,seed,file_header)
if "coeff" in args.analysis_options:
    compare_coefficients(test_functions,exactness_list,grid_lists,
                         file_header)
if "indexes" in args.analysis_options:
    compare_grid_indexes(test_dimensions,exactness_list,
                         grid_lists,file_header)

if "gridplot" in args.analysis_options:
    compare_grid_plot(exactness_list,grid_lists,file_header)
    
if "surrogateplot" in args.analysis_options:
    plot_funct = [x for x in test_functions if x.dim == 2]
    compare_surrogate_plot(plot_funct,exactness_list,points_plot,
                           grid_lists,file_header)
print("All requested files created.")