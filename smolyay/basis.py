import abc
import math

import numpy

class BasisFunction(abc.ABC):
    """Basis function for interpolating data.

     A one-dimensional basis function is defined on the domain
     :math:`[-1,1]`. The function defines the :attr:`points` at
     which it should be sampled within this interval for interpolation.
     The function also has an associated :meth:`__call__` method
     for evaluating it at a point within its domain.


    """

    def __init__(self):
        self._points = []

    @property
    @abc.abstractmethod
    def points(self):
        """list: Sampling points for interpolation."""
        pass


    @abc.abstractmethod
    def __call__(self,x):
        """Evaluate the basis function.

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
            self._points = list(numpy.polynomial.chebyshev.chebpts2(self._n+1))

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

    @classmethod
    def make_nested_set(cls, exactness):
        """calculate nested points for Chebyshev polynomial basis function
        Created the NestedBasisFunctionSet object that holds the nested
        points for a CHebyshev basis function

        Parameters
        ----------
        exactness : int
            Level of exactness to calculate points to.

        Returns
        -------
        NestedBasisFunctionSet object
            Data structure for points.
        """
        levels = []
        points = []

        end_level_index = [2**n for n in range(exactness+1)]
        end_level_index[0] = 0
        start_level_index = [0]
        start_level_index.extend([n+1 for n in end_level_index])
        max_degree = end_level_index[-1]
        basis_functions = [ChebyshevFirstKind(n) for n in range(max_degree+1)]

        for i in range(0,exactness+1):
            levels.append(
                    list(range(start_level_index[i],end_level_index[i]+1)))
            new_points = basis_functions[end_level_index[i]].points
            for p in new_points:
                if not numpy.isclose(points,p).any():
                    points.append(p)
        return NestedBasisFunctionSet(points,basis_functions,levels)


class BasisFunctionSet():
    """Set of basis functions
    Creates a framework for a set of basis functions to be used as the
    building blocks for a surrogate model.
    ``basis_functions`` is a list of BasisFunction objects
    ``points`` 1D points taken from the BasisFunction objects

    """

    def __init__(self,points,basis_functions):
        self._basis_functions = basis_functions
        self._points = points
        if len(basis_functions) != len(points):
            raise IndexError("basis_functions and points must have the "
                    "same number of elements.")

    @property
    def points(self):
        """All points for basis function set at some level of precision"""
        return self._points

    @property
    def basis_functions(self):
        """list of BasisFunction objects"""
        return self._basis_functions


class NestedBasisFunctionSet(BasisFunctionSet):
    """Set of basis functions that are nested

    The set of basis functions this class uses are assumed to be part
    of a sequential series of functions, where the functions that
    determine the points are nested
    ``levels`` is a list of lists that determines which grid level each
    point is added on
    """

    def __init__(self,points,basis_functions,levels):
        super().__init__(points,basis_functions)
        self._levels = levels

    @property
    def levels(self):
        """list of list that show the points for each grid level"""
        return self._levels

    @levels.setter
    def levels(self,levels):
        self._levels = levels

