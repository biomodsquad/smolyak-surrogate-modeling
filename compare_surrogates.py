import itertools
import time
import warnings

import numpy
import pandas
import scipy.stats

from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator)
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *


def compare_error(test_functions,exact_list,points_compare,
        grid_list_1,grid_list_1_name,grid_list_2,grid_list_2_name,file_header):
    '''Compares surrogates functions formed from different grid objects

    This analysis function takes in two lists of IndexGridGenerator objects
    and compares the error of the surrogates functions that can be made from
    them.

    Parameters
    ----------
    test_functions : list of objects class test_fun
        functions the surrogates will be fitted to

    exact_list : list of int
        levels of exactness used to create the grids

    points_compare : int
        number of points to be used to compare surrogates to the real function

    grid_list_1 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    grid_list_2 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file

    '''
    start_time = time.time()
    print('\nStart...')
    warnings.filterwarnings("error")
    # initialize information for files and DataFrames
    index_names = [f.name for f in test_functions]
    head_names = [grid_list_1_name+' µ=',grid_list_2_name+' µ=','Runtime µ=']
    column_names = [''.join(x) for x in 
        list(itertools.product(head_names,map(str,exact_list)))]
    start_time = time.time()
    ## Begin calculations
    print('Begin surrogate and error calculations.')
    data_collection = numpy.zeros((len(test_functions),len(exact_list)*3))

    try:
        for (func,j) in zip(test_functions,range(len(test_functions))):
            print('Estimating function : ' + func.name)
            func_time_start = time.time() # time each function
            # create test points
            points_gener = scipy.stats.qmc.LatinHypercube(d=
                func.dim).random(n=points_compare)
            test_points = scipy.stats.qmc.scale(points_gener, 
                func.lower_bounds, func.upper_bounds)
            real_output = [func(x) for x in test_points]
            # calculate surrogates for each exactness
            for (grid_1,grid_2,k) in zip(grid_list_1,grid_list_1,
                                               range(len(exact_list))):
                calc_time = time.time() # time each exactness
                ## Make Surrogate Models
                # initialize surrogates
                surrogate_1 = Surrogate(func.bounds,grid_1)
                surrogate_2 = Surrogate(func.bounds,grid_2)
                # get samples to train models
                data_1 = [func(point) for point in surrogate_1.points]
                data_2 = [func(point) for point in surrogate_2.points]
                # train models
                surrogate_1.train_from_data(data_1)
                surrogate_2.train_from_data(data_2)
                # test the error of surrogates
                error_1 = 0
                error_2 = 0
                # get mean squared error
                try:
                    for x,y in zip(test_points,real_output):
                        error_1 += (y - surrogate_1(x))**2
                        error_2 += (y - surrogate_2(x))**2
                    error_1 = error_1/float(points_compare)
                    error_2 = error_2/float(points_compare)
                except RuntimeWarning:
                    error_1 = -1
                    error_2 = -1
                calc_time_end = time.time() - calc_time # record time
                # collect data
                data_collection[j,k] = error_1
                data_collection[j,k+len(exact_list)] = error_2
                data_collection[j,k+2*len(exact_list)] = calc_time_end
            print('  Runtime: ' +str(round(time.time()-func_time_start,2))+' sec')
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
        ## Export data to file
        print('Finished calculations.')
        # print total time spent
        time_taken(start_time)
        # create DataFrame
        print('Creating File...')
        results = pandas.DataFrame(data_collection,
                                   index=index_names,columns=column_names)
        # export DataFrame
        file_name = file_header + '_points'+ str(points_compare) + '_error_results.csv'
        results.to_csv(file_name)
        print('File created.')
   
def compare_coefficients(test_functions,exact_list,
        grid_list_1,grid_list_1_name,grid_list_2,grid_list_2_name,file_header):
    '''Compares surrogates functions formed from different grid objects

    This analysis function takes in two lists of IndexGridGenerator objects,
    creates surrogate functions using the objects in each list, and then
    prints the coefficients for each function to a file

    Parameters
    ----------
    test_functions : list of objects class test_fun
        functions the surrogates will be fitted to

    exact_list : list of int
        levels of exactness used to create the grids

    grid_list_1 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    grid_list_2 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file

    '''
    start_time = time.time()
    print('\nStart...')
    warnings.filterwarnings("error")
    # initialize information for files and DataFrames
    head_names = [grid_list_1_name+' µ=',grid_list_2_name+' µ=']
    start_time = time.time()
    writer = pandas.ExcelWriter(file_header + '_coefficents.xlsx')
    ## Begin calculations
    print('Begin surrogate and error calculations.')

    try:
        for (func,j) in zip(test_functions,range(len(test_functions))):
            print('Estimating function : ' + func.name)
            func_time_start = time.time() # time each function
            # initialize container for coefficients
            coeff_data = {}
            ## Make Surrogate Models
            for (grid_1,grid_2,k) in zip(grid_list_1,grid_list_1,
                                         range(len(exact_list))):
                # initialize surrogates
                surrogate_1 = Surrogate(func.bounds,grid_1)
                surrogate_2 = Surrogate(func.bounds,grid_2)
                # get samples to train models
                data_1 = [func(point) for point in surrogate_1.points]
                data_2 = [func(point) for point in surrogate_2.points]
                # train models
                surrogate_1.train_from_data(data_1)
                surrogate_2.train_from_data(data_2)
                # collect surrogate data
                coeff_data[head_names[0]+
                                str(exact_list[k])] = surrogate_1.coefficients
                coeff_data[head_names[1]+
                                str(exact_list[k])] = surrogate_2.coefficients
            # add to excel sheet
            coeff_result = pandas.DataFrame.from_dict(coeff_data,orient='index')
            coeff_result = coeff_result.transpose()
            coeff_result.to_excel(writer,index=False,sheet_name=func.name)
            print('  Runtime: ' +str(round(time.time()-func_time_start,2))+' sec')
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
        ## Export data to file
        writer.close()
        print('Finished calculations.')
        # print total time spent
        time_taken(start_time)

def compare_grid_indexes(dimension_list,exact_list,
        grid_list_1,grid_list_1_name,grid_list_2,grid_list_2_name,file_header):
    '''Compare grid indexes for multiple dimensions

    This analysis function takes in two lists of IndexGridGenerator objects,
    and prints the indexes for each term in a surrogate function that would
    be generated by each object given some dimension

    Parameters
    ----------
    dimension_list : list of int
        dimensions to be analyzed

    exact_list : list of int
        levels of exactness used to create the grids

    grid_list_1 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    grid_list_2 : list of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    grid_list_1_name : string
        label for data from surrogates made by grid_list_1 in exported file

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file
    '''
    start_time = time.time()
    head_names = [grid_list_1_name+' µ=',grid_list_2_name+' µ=']
    grid_columns = [''.join(x)
                    for x in list(itertools.product(head_names,map(str,exact_list)))]
    writer = pandas.ExcelWriter(file_header + '_grids.xlsx')
    # Start adding to file
    try:
    grid_index_data = {}
    for d in dimension_list:
        grid_index_data = {}
        for i in range(len(exact_list)):
            grid_index_data[head_names[0]+
                            str(exact_list[i])] = grid_list_1[i](d).indexes
            grid_index_data[head_names[1]+
                            str(exact_list[i])] = grid_list_2[i](d).indexes
        grid_results = pandas.DataFrame.from_dict(grid_index_data,orient='index')
        grid_results = grid_results.transpose()
        grid_results.to_excel(writer,index=False,sheet_name='Dim = '+str(d))
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
    writer.close()
    print('Finished calculations.')
    # print total time spent
    time_taken(start_time)

def time_taken(start_time):
    '''Print out the time between the starting time and now

    Parameters
    ----------
    start_time : float
        the initial time
    '''
    end_time = time.time()
    total_time = end_time-start_time # time to finish executing
    print('Time to calculate: ',end='')
    if (total_time > 3600):
        print(str(total_time//3600) +' hr ' +
              str((total_time % 3600)//60) + ' min ' +
              str((total_time % 3600) % 60) + ' sec')
    elif(total_time > 60):
        print(str(total_time//60) +' min ' + str(total_time % 60) + ' sec')
    else:
        print(str(total_time) + ' sec')
    print('Time to calculate: ' +  str(total_time) + ' sec')
    print('')
