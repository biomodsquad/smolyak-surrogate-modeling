import pytest
import numpy
from smolyay.smolyak import IndexGrid


grid_points_expect = numpy.array([[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,0],[4,0]], dtype=numpy.int32)


def test_initial_dimension():
    """test initialized class returns correct dimension"""
    test_class = IndexGrid(1,2,[1,2,3])
    assert test_class.dimension == 1

def test_initial_exactness():
    """test initialized class returns correct exactness"""
    test_class = IndexGrid(1,2,[1,2,3])
    assert test_class.exactness == 2

def test_initial_index_by_level():
    """test initialized class returns correct index by level"""
    test_class = IndexGrid(1,2,[1,2,3])
    assert test_class.index_per_level == [1,2,3]

def test_level_indexes():
    """test level indexes"""
    test_class = IndexGrid(2,2,[1,2,2])
    level_indexes_expect = [[0],[1,2],[3,4]]
    level_indexes_recieved = test_class.level_indexes
    error_message = []
    is_equal = True
    # check length, then values
    if len(level_indexes_recieved) == 3:
        counter = 0
        while is_equal and counter < 3:
            is_equal = is_equal and numpy.array_equal(
                    level_indexes_expect[counter],
                    level_indexes_recieved[counter])
            if not is_equal:
                error_message.append(
                        "level indexes incorrect at level" + str(counter))
            counter += 1
    else:
        error_message.append("level_indexes length incorrect")

    assert not error_message, error_message

def test_grid_point_index():
    """test grid point index"""
    test_class = IndexGrid(2,2,[1,2,2])
    grid_points = test_class.grid_point_index
    # sort grid_points as order of points is unimportant
    for i in range(len(grid_points[0,:])-1,-1,-1):
        grid_points = grid_points[grid_points[:,i].argsort(kind='mergesort')]

    assert numpy.array_equal(grid_points,grid_points_expect)

def test_level_indexes_extra_level_per_index():
    """test level indexes when level_per_index has more information"""
    test_class = IndexGrid(2,2,[1,2,2,4,8])
    level_indexes_expect = [[0],[1,2],[3,4]]
    level_indexes_recieved = test_class.level_indexes
    error_message = []
    is_equal = True
    # check length, then values
    if len(level_indexes_recieved) == 3:
        counter = 0
        while is_equal and counter < 3:
            is_equal = is_equal and numpy.array_equal(
                    level_indexes_expect[counter],
                    level_indexes_recieved[counter])
            if not is_equal:
                error_message.append(
                        "level indexes incorrect at level" + str(counter))
            counter += 1
    else:
        error_message.append("level_indexes length incorrect")

    assert not error_message, error_message

def test_grid_point_index_extra_level_per_index():
    """test grid point index when level_per_index has more information"""
    test_class = IndexGrid(2,2,[1,2,2,4,8])
    grid_points = test_class.grid_point_index
    # sort grid_points as order of points is unimportant
    for i in range(len(grid_points[0,:])-1,-1,-1):
        grid_points = grid_points[grid_points[:,i].argsort(kind='mergesort')]

    assert numpy.array_equal(grid_points,grid_points_expect)


def test_index_per_level_expand():
    """test if properties change if more is added to level_per_index"""
    test_class = IndexGrid(1,2,[1,2,3])
    level_indexes_1 = test_class.level_indexes
    grid_point_1 = test_class.grid_point_index

    test_class.index_per_level = [1,2,3,4,5]
    level_indexes_2 = test_class.level_indexes
    grid_point_2 = test_class.grid_point_index
    
    assert numpy.array_equal(grid_point_1,grid_point_2)

def test_update_exactness():
    """test if grid points are correct if exactness is updated"""
    test_class = IndexGrid(2,1,[1,2,2])
    grid_points_1 = test_class.grid_point_index
    test_class.exactness = 2
    grid_points_2 = test_class.grid_point_index
    test_class.exactness = 1
    grid_points_3 = test_class.grid_point_index
    #sort grid_points as order is unimportant
    for i in range(len(grid_points_2[0,:])-1,-1,-1):
        grid_points_1 = grid_points_1[
                grid_points_1[:,i].argsort(kind='mergesort')]
        grid_points_2 = grid_points_2[
                grid_points_2[:,i].argsort(kind='mergesort')]
        grid_points_3 = grid_points_3[
                grid_points_3[:,i].argsort(kind='mergesort')]
    #see if test failed
    exactness_2e = numpy.array_equal(grid_points_2, grid_points_expect)
    exactness_13 = numpy.array_equal(grid_points_1, grid_points_3)
    exactness_12 = numpy.array_equal(grid_points_1, grid_points_2)
    exactness_23 = numpy.array_equal(grid_points_3, grid_points_2)
    #include ways test failed if it did
    error_message = []
    if exactness_12:
        error_message.append("Increasing exactness failed to update value. ")
    elif not exactness_2e:
        error_message.append("Increased exactness has incorrect value. ")
    if exactness_23 and not exactness_12:
        error_message.append("Decreasing exactness failed to update value. ")
    elif not exactness_13:
        error_message.append("Decreased exactness has incorrect value. ")


    assert not error_message,''.join(error_message)

def test_update_dimension():
    """test if grid points are correct if dimension is updated"""
    test_class = IndexGrid(3,2,[1,2,2])
    grid_points_1 = test_class.grid_point_index
    test_class.dimension = 2
    grid_points_2 = test_class.grid_point_index
    test_class.dimension = 3
    grid_points_3 = test_class.grid_point_index
    #sort grid_points as order is unimportant
    for i in range(len(grid_points_2[0,:])-1,-1,-1):
        grid_points_1 = grid_points_1[
                grid_points_1[:,i].argsort(kind='mergesort')]
        grid_points_2 = grid_points_2[
                grid_points_2[:,i].argsort(kind='mergesort')]
        grid_points_3 = grid_points_3[
                grid_points_3[:,i].argsort(kind='mergesort')]
    #see if test failed
    dimension_2e = numpy.array_equal(grid_points_2, grid_points_expect)
    dimension_13 = numpy.array_equal(grid_points_1, grid_points_3)
    dimension_12 = numpy.array_equal(grid_points_1, grid_points_2)
    dimension_23 = numpy.array_equal(grid_points_3, grid_points_2)
    #include ways test failed if it did
    error_message = []
    if dimension_12:
        error_message.append("Decreasing dimension failed to update value. ")
    elif not dimension_2e:
        error_message.append("Decreased dimension has incorrect value. ")
    if dimension_23 and not dimension_12:
        error_message.append("Increasing dimension failed to update value. ")
    elif not dimension_13:
        error_message.append("Increased dimension has incorrect value. ")

    assert not error_message,''.join(error_message)

def test_update_index_per_level():
    """test if grid points are correct if index_per_level is updated"""
    test_class = IndexGrid(2,2,[1,2,3])
    grid_points_1 = test_class.grid_point_index
    test_class.index_per_level = [1,2,2]
    grid_points_2 = test_class.grid_point_index
    #sort grid_points as order is unimportant
    for i in range(len(grid_points_2[0,:])-1,-1,-1):
        grid_points_1 = grid_points_1[
                grid_points_1[:,i].argsort(kind='mergesort')]
        grid_points_2 = grid_points_2[
                grid_points_2[:,i].argsort(kind='mergesort')]
    #see if test failed
    lpi_2e = numpy.array_equal(grid_points_2, grid_points_expect)
    lpi_12 = numpy.array_equal(grid_points_1, grid_points_2)
    #include ways test failed if it did
    error_message = []
    if lpi_12:
        error_message = "New index per level failed to update value. "
    elif not lpi_2e:
        error_message = "New index per level has incorrect value. "

    assert not error_message,error_message

def test_error():
    """test if error is returned for invalid index_per_level"""
    test_class = IndexGrid(2,2,[1,2])
    with pytest.raises(IndexError):
        level_indexes = test_class.level_indexes
