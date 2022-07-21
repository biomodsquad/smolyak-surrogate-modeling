import pytest
from basis.ChebyshevFirstKind import ChebyshevFirstKind
from basis.BasisFunction import BasisFunction

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

def test_basis_degree_0():
    """Chebyshev polynomial degree 0 is 1"""
    for i in range(0,16):
        assert ChebyshevFirstKind.basis(i,0) == 1

def test_basis_degree_1():
    """Chebyshev polynomial degree 1 should return input"""
    for i in range(0,16):
        assert ChebyshevFirstKind.basis(i,1) == i

def test_basis_input_1():
    """Chebyshev polynomial input 1 returns 1 for any degree n"""
    for i in range(0,16):
        assert ChebyshevFirstKind.basis(1,i) == 1

def test_basis_input_2_degree_6():
    """Test chebyshev polynomial at some degree at some input"""
    assert ChebyshevFirstKind.basis(2,6) == 1351

def test_is_abstract():
    """Check BasisFunction is an abstract class"""
    with pytest.raises(TypeError):
        test_class = BasisFunction(1)


