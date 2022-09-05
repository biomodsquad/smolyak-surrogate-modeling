import abc
import math

import numpy

class BasisFunction(abc.ABC):
    """Abstract class for basis functions.
    Data structure for basis functions that can be called as a function
    and holds the points 
    ``points`` property represents all the unique 1D points within the 
    range [-1,1] of the basis function associated with a Smolyak index 
    described by the class IndexGrid.

    """

    def __init__(self):
        self._points = []

    @abc.abstractproperty
    def points(self):
        """The points of the basis function assigned to Smolyak indices"""
        pass


    @abc.abstractmethod
    def __call__(self,x):
        """Compute term of basis function
        Returns the output of the basis function with input x

        Parameters
        ----------
        x : float
            Input arguement.

        Returns
        -------
        float
            Value of basis function.
        """
        pass


class ChebyshevFirstKind(BasisFunction):
    r"""Basis function Chebyshev polynomials of the first kind
    Chebyshev polynomials are a sequence of polynomials described by the
    following recurrence relation:

    ..math:
    T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)

    Object describes the nth term in the set of polynomials as designated
    by the property ``n``
    ``points`` represents the extrema of the polynomial of this degree on
    the domain [-1,1]

    The extrema of the polynomial are calculated via the following
    equation:

    ..math:
        x_{i}^* = -\cos\left((\frac{i-1}{n-1}\pi\right), i = 1,...,n

    Parameters
    ----------
    n : int
        Degree of the Chebyshev polynomial.
    """

    def __init__(self,n):
        super().__init__()
        self._n = n

    @property
    def points(self):
        """extrema of polynomial"""
        if len(self._points) == 0:
            self._compute_points()
        return self._points

    @property
    def n(self):
        """degree of polynomial"""
        return self._n

    def _compute_points(self):
        """Compute extrema of Chebyshev polynomial of the first kind"""
        if self._n == 0:
            self._points = [0]
        else:
            i = numpy.arange(self._n+1)
            self._points = list(-numpy.cos(numpy.pi*i/self._n))

    def __call__(self,x):
        """Terms of basis function
        Returns the output of an nth degree Chebyshev polynomial of the
        first kind with input x

        Parameters
        ----------
        x : float
            Input arguement.

        Returns
        -------
        float
            Value of Chebyshev polynomial of the first kind.
        """
        if self._n == 0:
            return 1
        elif self._n == 1:
            return x
        else:
            k_lim = self._n//2
            answer = 0
            for k in range(0,k_lim+1):
                answer += math.comb(self._n,2*k)*((x**2 - 1)**k)*(x**(self._n-2*k))
            return answer


class BasisFunctionSet():
    """Set of basis functions
    Creates a framework for a set of basis functions to be used as the
    building blocks for a surrogate model.
    ``basis_set`` is a list of BasisFunction objects
    ``points`` 1D points taken from the BasisFunction objects

    """

    def __init__(self,points,basis_set):
        self._basis_set = basis_set
        self._points = points

    @property
    def points(self):
        """All points for basis function set at some level of precision"""
        return self._points

    @property
    def basis_set(self):
        """list of BasisFunction objects"""
        return self._basis_set


class NestedBasisFunctionSet(BasisFunctionSet):
    """Set of basis functions that are nested

    The set of basis functions this class uses are assumed to be part
    of a sequential series of functions, where the functions that
    determine the points are nested
    ``levels`` is a list of lists that determines which grid level each
    point is added on
    """

    def __init__(self,points,basis_set,levels):
        """Initialization of parameters"""
        super().__init__(points,basis_set)
        self._levels = levels

    @property
    def levels(self):
        """list of list that show the points for each grid level"""
        return self._levels

    @levels.setter
    def levels(self,levels):
        self._levels = levels


def make_nested_chebyshev_points(exactness,basis_function):
    """calculate nested points for Chebyshev polynomial basis function
    Created the NestedBasisFunctionSet object that holds the nested
    points for a CHebyshev basis function

    Parameters
    ----------
    exactness : int
        Level of exactness to calculate points to.
    basis_function : BasisFunction object
        Type of chebyshev polynomial.

    Returns
    -------
    NestedBasisFunctionSet object
        Data structure for points.
    """

    levels = [[0]]
    basis_set = []
    points = [0]
    max_degree = 2**exactness
    if exactness == 0:
        max_degree = 0
    start_level_index = 1
    for i in range(0,max_degree+1):
        basis_set.append(basis_function(i))

    for j in range(1,exactness+1):
        degree_sample = 2**j
        new_points = basis_set[degree_sample].points
        end_level_index = start_level_index
        for k in range(0,len(new_points)):
            if not numpy.isclose(points,new_points[k]).any():
                points.append(new_points[k])
                end_level_index = end_level_index + 1
        levels.append(list(range(start_level_index,end_level_index)))
        start_level_index = end_level_index

    return NestedBasisFunctionSet(points,basis_set,levels)

