import pytest
import numpy
from smolyay.smolyak import IndexGrid,generate_compositions

@pytest.fixture
def expected_points():
    """grid_point_indexes for dimension=2,exactness=2,index_per_level=[1,2,2]"""
    return numpy.array([[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],
        [1,1],[1,2],[2,0],[2,1],[2,2],[3,0],[4,0]], dtype=numpy.int32)

@pytest.fixture
def expected_indexes():
    """level_indexes for dimension=2,exactness=2,index_per_level=[1,2,2]"""
    return [[0],[1,2],[3,4]]

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
    assert numpy.array_equiv(test_class.index_per_level,[1,2,3])

def test_level_indexes(expected_indexes):
    """test level indexes"""
    test_class = IndexGrid(2,2,[1,2,2])
    level_indexes_recieved = test_class.level_indexes
    
    assert level_indexes_recieved == expected_indexes

def test_grid_point_index(expected_points):
    """test grid point index"""
    test_class = IndexGrid(2,2,[1,2,2])
    grid_points = sorted(test_class.grid_point_indexes,key = lambda x:x[0])

    assert numpy.array_equal(grid_points,expected_points)

def test_level_indexes_extra_level_per_index(expected_indexes):
    """test level indexes when level_per_index has more information"""
    test_class = IndexGrid(2,2,[1,2,2,4,8])
    level_indexes_recieved = test_class.level_indexes

    assert level_indexes_recieved == expected_indexes

def test_grid_point_index_extra_level_per_index(expected_points):
    """test grid point index when level_per_index has more information"""
    test_class = IndexGrid(2,2,[1,2,2,4,8])
    grid_points = sorted(test_class.grid_point_indexes,key = lambda x:x[0])

    assert numpy.array_equal(grid_points,expected_points)

def test_index_per_level_expand():
    """test if properties change if more is added to level_per_index"""
    test_class = IndexGrid(1,2,[1,2,3])
    grid_point_1 = test_class.grid_point_indexes

    test_class.index_per_level = [1,2,3,4,5]
    grid_point_2 = test_class.grid_point_indexes
    
    assert numpy.array_equal(grid_point_1,grid_point_2)

def test_increase_exactness(expected_points):
    """test if grid points are correct if exactness is increased"""
    test_class = IndexGrid(2,1,[1,2,2])
    grid_points_1 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    test_class.exactness = 2
    grid_points_2 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    assert ((not numpy.array_equiv(grid_points_1,grid_points_2))
            and numpy.array_equiv(grid_points_2, expected_points))

def test_decrease_exactness(expected_points):
    """test if grid points are correct if exactness is decreased"""
    test_class = IndexGrid(2,4,[1,2,2,4,8])
    grid_points_1 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    test_class.exactness = 2
    grid_points_2 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    assert ((not numpy.array_equiv(grid_points_1,grid_points_2))
            and numpy.array_equiv(grid_points_2, expected_points))

def test_increase_dimension(expected_points):
    """test if grid points are correct if dimension is increased"""
    test_class = IndexGrid(1,2,[1,2,2])
    grid_points_1 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    test_class.dimension = 2
    grid_points_2 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    assert ((not numpy.array_equiv(grid_points_1,grid_points_2))
            and numpy.array_equiv(grid_points_2, expected_points))

def test_decrease_dimension(expected_points):
    """test if grid points are correct if dimension is decreased"""
    test_class = IndexGrid(3,2,[1,2,2])
    grid_points_1 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    test_class.dimension = 2
    grid_points_2 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    assert ((not numpy.array_equiv(grid_points_1,grid_points_2))
            and numpy.array_equiv(grid_points_2, expected_points))

def test_update_index_per_level(expected_points):
    """test if grid points are correct if index_per_level is changed"""
    test_class = IndexGrid(2,2,[1,2,3])
    grid_points_1 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    test_class.index_per_level = [1,2,2]
    grid_points_2 = sorted(test_class.grid_point_indexes,key = lambda x:x[0])
    assert ((not numpy.array_equiv(grid_points_1,grid_points_2))
            and numpy.array_equiv(grid_points_2, expected_points))

def test_invalid_index_per_level():
    """test if error is returned if invalid index_per_level is initialized"""
    test_class = IndexGrid(2,2,[1,2])
    with pytest.raises(IndexError):
        level_indexes = test_class.level_indexes

def test_update_invalid_index_per_level():
    """test if error is returned if updated index_per_level is invalid"""
    test_class = IndexGrid(2,2,[1,2,2])
    test_class.index_per_level = [2,2]
    with pytest.raises(IndexError):
        level_indexes = test_class.level_indexes

def test_exactness_zero_level_indexes():
    """test the level indexes if exactness is zero"""
    test_class = IndexGrid(2,0,[1,2,2])
    assert test_class.level_indexes == [[0]]

def test_exactness_zero_grid_point_indexes():
    """test the level indexes if exactness is zero"""
    test_class = IndexGrid(2,0,[1,2,2])
    assert numpy.array_equiv(test_class.level_indexes,[0,0])

def test_generate_compositions_include_zero_true():
    """test the generate compositions function if include_zero is true"""
    composition_expected = numpy.array([[6,0],[5,1],[4,2],[3,3],
        [2,4],[1,5],[0,6]],dtype=numpy.int32)
    composition_obtained = []
    for i in generate_compositions(6,2,include_zero=True):
        composition_obtained.append(i.copy())
    assert numpy.array_equiv(composition_obtained,composition_expected)

def test_generate_compositions_include_zero_false():
    """test the generate compositions function if include_zero is false"""
    test_class = IndexGrid(2,1,[1,2,3])
    composition_expected = numpy.array([[5,1],[4,2],[3,3],
        [2,4],[1,5]],dtype=numpy.int32)
    composition_obtained = list(generate_compositions(6,2,include_zero=False))
    assert numpy.array_equiv(composition_obtained,composition_expected) 

def test_generate_compositions_zero_false_error():
    """test that generate compositions raises an error for invalid input"""
    with pytest.raises(ValueError):
        composition_obtained = generate_compositions(6,7,include_zero=False)
        for obj in composition_obtained:
            pass
