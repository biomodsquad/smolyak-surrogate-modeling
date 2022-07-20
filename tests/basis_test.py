import pytest
from basis.ChebyshevFirstKind import ChebyshevFirstKind

@pytest.fixture
def expected_extrema_2():
    """extrema for exactness = 2"""
    return [0, -1.0, 1.0, -0.707106781186548, 0.707106781186548]

@pytest.fixture
def expected_extrema_3():
    """extrema for exactness = 3"""
    return [0, -1.0, 1.0, -0.707106781186548, 0.707106781186548, 
            -0.923879532511287, -0.38268343236509, 0.38268343236509, 
            0.923879532511287]

@pytest.fixture
def expected_extrema_4():
    """extrema for exactness = 4"""
    return [0, -1.0, 1.0, -0.707106781186548, 0.707106781186548,
            -0.923879532511287, -0.38268343236509, 0.38268343236509, 
            0.923879532511287, -0.98078528040323, -0.831469612302545, 
            -0.555570233019602, -0.195090322016128, 0.195090322016128, 
            0.555570233019602, 0.831469612302545, 0.98078528040323]

def test_exactness_zero():
    """test exactness of zero"""
    test_class = ChebyshevFirstKind(0)
    assert test_class.max_exactness == 0
    assert test_class.extrema == [0]
    assert test_class.extrema_per_level == [[0]]
    assert test_class.num_extrema_per_level == [1]

def test_initial_exactness_1():
    """test initial when exactness is 1"""
    test_class = ChebyshevFirstKind(1)
    assert test_class.max_exactness == 1
    assert test_class.extrema == [0,-1,1]
    assert test_class.extrema_per_level == [[0],[1,2]]
    assert test_class.num_extrema_per_level == [1,2]

def test_initial_exactness_2(expected_extrema_2):
    """test initial when exactness is 2"""
    test_class = ChebyshevFirstKind(2)
    assert test_class.max_exactness == 2
    assert test_class.extrema == expected_extrema_2
    assert test_class.extrema_per_level == [[0],[1,2],[3,4]]
    assert test_class.num_extrema_per_level == [1,2,2]

def test_increase_exactness(expected_extrema_4):
    """test when max exactness is increased"""
    test_class = ChebyshevFirstKind(2)
    test_class.max_exactness = 4
    assert test_class.max_exactness == 4
    assert test_class.extrema == expected_extrema_4
    assert test_class.extrema_per_level == [[0],[1,2],[3,4],[5,6,7,8],
            [9,10,11,12,13,14,15,16]]
    assert test_class.num_extrema_per_level == [1,2,2,4,8]

def test_decrease_exactness(expected_extrema_3):
    """test when max exactness is decreased"""
    test_class = ChebyshevFirstKind(5)
    test_class.max_exactness = 3
    assert test_class.max_exactness == 3
    assert test_class.extrema == expected_extrema_3
    assert test_class.extrema_per_level == [[0],[1,2],[3,4],[5,6,7,8]]
    assert test_class.num_extrema_per_level == [1,2,2,4]

