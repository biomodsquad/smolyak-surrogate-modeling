import itertools
import time
import warnings

import matplotlib.pyplot
import numpy
import pandas
import scipy.stats

from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator)
from smolyay.surrogate import Surrogate
from smolyay.test_function_class import *


def compare_error(test_functions,exact_list,points_compare,grid_lists,
        file_header):
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

    grid_lists : dict of lists of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

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
    head_names = [x + ' µ =' for x in list(grid_lists.keys())]
    column_names = [''.join(x) for x in 
        list(itertools.product(head_names,map(str,exact_list)))]
    # Begin calculations
    print('Begin surrogate and error calculations.')
    error_collection = numpy.zeros((len(test_functions),len(column_names)))
    runtime_collection = numpy.zeros((len(test_functions),len(column_names)))
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
            for i in range(len(exact_list)):
                for (k,c) in zip(grid_lists.keys(),
                                 range(len(grid_lists.keys()))):
                    calc_time = time.time() # time each exactness
                    ## Make Surrogate Models
                    grid = grid_lists[k][i]
                    # initialize surrogates
                    surrogate = Surrogate(func.bounds,grid)
                    # get samples to train models
                    data = [func(point) for point in surrogate.points]
                    # train models
                    surrogate.train_from_data(data)
                    # test the error of surrogates
                    error = 0
                    # get mean squared error
                    try:
                        for x,y in zip(test_points,real_output):
                            error += (y - surrogate(x))**2
                        error = error/float(points_compare)
                    except RuntimeWarning:
                        error = -1
                    calc_time_end = time.time() - calc_time # record time
                    # collect data
                    error_collection[j,i+c*len(exact_list)] = error
                    runtime_collection[j,i+c*len(exact_list)] = calc_time_end
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
        error_results = pandas.DataFrame(error_collection,
                                   index=index_names,columns=column_names)
        runtime_results = pandas.DataFrame(runtime_collection,
                                   index=index_names,columns=column_names)
        # export DataFrame
        file_name = (file_header + '_points'+ str(points_compare) +
                     '_error_results.xlsx')
        writer = pandas.ExcelWriter(file_name)
        error_results.to_excel(writer,index=False,sheet_name='Error')
        runtime_results.to_excel(writer,index=False,sheet_name='Runtime')
        writer.close()
        print('File created.')
   
def compare_coefficients(test_functions,exact_list,grid_lists,file_header):
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
    
    grid_lists : dict of lists of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

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
            for i in range(len(exact_list)):
                for k in grid_lists.keys():
                    # initialize surrogates
                    surrogate = Surrogate(func.bounds,grid_lists[k][i])
                    # get samples to train models
                    data = [func(point) for point in surrogate.points]
                    # train models
                    surrogate.train_from_data(data)
                    # collect surrogate data
                    coeff_data[k +' µ =' +
                            str(exact_list[i])] = surrogate.coefficients
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

def compare_grid_indexes(dimension_list,exact_list,grid_lists,file_header):
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

    grid_lists : dict of lists of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file
    '''
    start_time = time.time()
    writer = pandas.ExcelWriter(file_header + '_grids.xlsx')
    # Start adding to file
    try:
        grid_index_data = {}
        for d in dimension_list:
            grid_index_data = {}
            for i in range(len(exact_list)):
                for k in grid_lists.keys():
                    grid_index_data[k +' µ='+ str(exact_list[i])
                                    ] = grid_lists[k][i](d).indexes
            grid_results = pandas.DataFrame.from_dict(grid_index_data,
                    orient='index')
            grid_results = grid_results.transpose()
            grid_results.to_excel(writer,index=False,sheet_name='Dim = '+str(d))
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
        writer.close()
        print('Finished calculations.')
        # print total time spent
        time_taken(start_time)

def compare_grid_plot(exact_list,grid_lists,file_header):
    '''Compare grid indexes for multiple dimensions

    This analysis function takes in lists of IndexGridGenerator objects,
    and plots the 2D representation of each Smolyak Grid that would be
    generated in 2 dimensions

    Parameters
    ----------

    exact_list : list of int
        levels of exactness used to create the grids

    grid_lists : dict of lists of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file
    '''
    start_time = time.time()
    ax_rows = len(exact_list)
    ax_columns = len(grid_lists.keys())
    fig, ax = matplotlib.pyplot.subplots(ax_rows,ax_columns,
                                         subplot_kw=dict(box_aspect=1),
                                         figsize=(ax_columns*4,ax_rows*4))
    # Start adding to file
    try:
        for i in range(ax_rows):
            ax[i,0].set_ylabel('µ = ' + str(exact_list[i]))
        for k,j in zip(grid_lists.keys(),list(range(ax_columns))):
            ax[0,j].set_title(k)
            for i in range(ax_rows):
                ax[i,j].scatter(*numpy.array(grid_lists[k][i](2).points).T,
                                linewidth=0)
        matplotlib.pyplot.savefig(file_header + '_gridplot.png',
                                  bbox_inches='tight')
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
        print('Finished calculations.')
        # print total time spent
        time_taken(start_time)

def compare_surrogate_plot(test_functions,exact_list,points_compare,
                           grid_lists,file_header):
    '''Compare surrogate plots for 2D functions

    This analysis function takes in lists of IndexGridGenerator objects and a
    list of test functions with 2 variables and plots the surrogates in
    comparison with the real function.

    Parameters
    ----------

    test_functions : list of objects class test_fun
        functions the surrogates will be fitted to
    
    exact_list : list of int
        levels of exactness used to create the grids

    points_compare : int
        an approximate of the number of points used to graph the functions
        squared
        
    grid_lists : dict of lists of IndexGridGenerator objects
        grids for each exactness that will create surrogate functions

    file_header : string
        information to go at the beginning of the file name

    Returns
    -------
    file
    '''
    warnings.filterwarnings("error")
    start_time = time.time()
    ax_rows = len(exact_list)
    ax_columns = len(grid_lists.keys())
    fig, ax = matplotlib.pyplot.subplots(ax_rows,ax_columns+1,
                                         subplot_kw=dict(box_aspect=1),
                                         figsize=((ax_columns+1)*4,ax_rows*4))
    num_level = 20
    # check that all test functions have only 2 dimensions
    if not all(x.dim == 2 for x in test_functions):
        raise ValueError('test functions must have only 2 variables')
    points_compare = int(numpy.sqrt(points_compare))
    # Start creating plots
    try:
        for func in test_functions:
            # create axis
            fig, ax = matplotlib.pyplot.subplots(ax_rows,ax_columns+1,
                                         subplot_kw=dict(box_aspect=1),
                                         figsize=((ax_columns+1)*4,ax_rows*4))
            
            print('Plotting function : ' + func.name)
            func_time_start = time.time() # time each function
            # create grids
            X = numpy.linspace(*func.bounds[0],points_compare)
            Y = numpy.linspace(*func.bounds[1],points_compare)
            X_grid,Y_grid = numpy.meshgrid(X,Y)
            Z = numpy.zeros(X_grid.shape)
            # plot real function
            for m in range(0,points_compare):
                for n in range(0,points_compare):
                    Z[m,n] = func([X[n],Y[m]])
            print(Z)
            ax[0,ax_columns].set_title(func.name)
            p = ax[0,ax_columns].contourf(X_grid,Y_grid,Z,levels=num_level,
                                      cmap='YlGnBu')
            # trying to get all the plots to use the real function's scale
            fig.colorbar(p,ax=ax[0,ax_columns])
            vmin= p.zmin
            vmax = p.zmax
            # name rows
            for i in range(ax_rows):
                ax[i,0].set_ylabel('µ = ' + str(exact_list[i]))
            for k,j in zip(grid_lists.keys(),list(range(ax_columns))):
                ax[0,j].set_title(k) # title columns
                for i in range(ax_rows):
                    # create surrogates
                    grid = grid_lists[k][i]
                    surrogate = Surrogate(func.bounds,grid)
                    data = [func(point) for point in surrogate.points]
                    # train models
                    surrogate.train_from_data(data)
                    # get points to plot
                    try:
                        for m in range(0,points_compare):
                            for n in range(0,points_compare):
                                Z[m,n] = surrogate([X[n],Y[m]])
                        ax[i,j].contourf(X_grid,Y_grid,Z,levels=num_level,
                                         cmap='YlGnBu',vmin=vmin,vmax=vmax)
                    except RuntimeWarning:
                        pass
            # save image
            matplotlib.pyplot.savefig(file_header +'_'+func.name+
                                      '_gridplot.png',
                                      bbox_inches='tight')
            # reset axis
            matplotlib.pyplot.close(fig)
            print('  Runtime: ' +
                  str(round(time.time()-func_time_start,2))+' sec')
    except KeyboardInterrupt:
        print('Terminated prematurely.')
    finally:
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
