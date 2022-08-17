import abc
import math

import numpy

class BasisFunction(abc.ABC):
    """Abstract class for basis functions.

    Creates a framework for basis functions that are the building blocks
    for the terms in the surrogate model. 
    ``points`` property represents all the unique 1D points of the 
    basis function associated with a Smolyak index described by the 
    class IndexGrid.

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
            input

        Returns
        -------
        y : float
            output
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

    ``n`` is the degree of the polynomial this object describes
    ``points`` represents the extrema of the polynomial of this degree on
    the domain [-1,1]

    The extrema of the polynomial are calculated via the following
    equation:

    ..math:
        x_{n,j}^* = -\cos\left((\frac{j-1}{n-1}\pi\right), j = 1,...,n
        n = 2^{exactness-1}+1

    Parameters
    ----------
    n : int
        degree of the Chebyshev polynomial
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

    @n.setter
    def n(self,n):
        if self._n != n:
            self._n = n
            self._points = []

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
            input

        Returns
        -------
        y : float
            output
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


class BasisFunctionSet(abc.ABC):
    """Set of basis functions
    Creates a framework for a set of basis functions to be used as the
    building blocks for a surrogate model.
    ``basis_set`` is a list of BasisFunction objects
    ``sample_flag`` is a list of boolean equal in length to basis_set and
    an index in the list is true if the object in basis_set at the same
    index is one that points should be sampled from
    ''all_points'' 1D points taken from the BasisFunction objects specified
    by sample_flag
    ``need_update`` is a boolean parameter that specifies when properties
    of the class need to be recomputed

    """

    def __init__(self,sample_flag,basis_set):
        self._sample_flag = sample_flag
        self._basis_set = basis_set
        self._all_points = []
        self._need_update = True

    @abc.abstractproperty
    def all_points(self):
        """All points for basis function set at some level of precision"""
        pass

    @property
    def basis_set(self):
        """list of BasisFunction objects"""
        return self._basis_set

    @basis_set.setter
    def basis_set(self,basis_set):
        self._basis_set = basis_set
        self._need_update = True

    @property
    def sample_flag(self):
        """list of what objects in basis_set to sample points from"""
        return self._sample_flag

    @sample_flag.setter
    def sample_flag(self,sample_flag):
        self._sample_flag = sample_flag
        self._need_update = True

    @abc.abstractmethod
    def _update(self):
        """Updates all_points based on parameters"""
        pass


class RecurrenceSet(BasisFunctionSet):
    """Set of basis functions related via a recurrence relation

    The set of basis functions this class uses are assumed to be part
    of a sequential series of functions, where the index of a function
    in basis_set is the only required input of the basis_function. Such
    functions can be described as a recurrence relation where the nth
    function of the sequence is a combination of previous terms.
    """

    def __init__(self,sample_flag,basis_set):
        """Initialization of parameters"""
        super().__init__(sample_flag,basis_set)
        self._update()

    @property
    def all_points(self):
        """All points for basis function set at some level of precision"""
        if self._need_update:
            self._update()
        return self._all_points

    def _update(self):
        """Update basis_set and get points"""
        self._all_points = []

        for i in range(0,len(self._sample_flag)):
            if self._sample_flag[i]:
                new_points = self._basis_set[i].points
                for j in range(0,len(new_points)):
                    if not numpy.isclose(self._all_points,new_points[j]).any():
                        self._all_points.append(new_points[j])
        self._need_update = False

    def __call__(self,n,x):
        """Compute value of function in set
        Computes the answer of the nth function of the set given input x

        Parameters
        ----------
        x : float
            input
        n : int
            index of function in set
        
        Returns
        -------
        answer : float
            output
        """
        return self._basis_set[n](x) 
