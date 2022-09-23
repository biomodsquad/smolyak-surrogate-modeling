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
            One-dimensional point.

        Returns
        -------
        float
            Value of basis function.
        """
        pass


class ChebyshevFirstKind(BasisFunction):
    r"""Chebyshev polynomial of the first kind.

    The Chebyshev polynomial of degree *n* is defined by the
    recursive relationship:

    ..math::

        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

    The :attr:`points` for this polynomial are the extrema on the domain
    :math:`[-1,1]`:
   
    ..math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

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
        r"""Evaluate the basis function.

        The Chebyshev polynomial is evaluated using the combinatorial formula:
        
        .. math::
        
            T_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} {n \ choose 2k} (x^2-1)^k x^{n-2k}

        for :math:`n \ge 2`.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[-1,1]`.

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
        """Create a nested set of Chebyshev polynomials.

        A nested set is created up to a given level of ``exactness``,
        which corresponds to a highest-order Chebyshev polynomial of
        degree ``n = 2**exactness``.

        Each nesting level corresponds to the increasing powers of 2 going up to
        ``2**exactness``, with the first level being a special case. The generating
        Chebyshev polynomials are hence of degree (0, 2, 4, 8, ...). Each new point
        added in a level is paired with a basis function of increasing order.

        For example, for an ``exactness`` of 3, the generating polynomials are
        of degree 0, 2, 4, and 8, at each of 4 levels. There are 1, 2, 2, and 4
        new points added at each level. The polynomial at level 0 is of degree 0,
        the polynomials at level 1 are of degrees 1 and 2, those at level 2 are of
        degree 3 and 4, and those at level 3 are of degrees 5, 6, 7, and 8.

        Parameters
        ----------
        exactness : int
            Level of exactness.

        Returns
        -------
        NestedBasisFunctionSet
            Nested Chebyshev polynomials of the first kind.

        """
        basis_functions = []
        levels = []
        points = []
        for i in range(0, exactness+1):
            if i > 1:
                start_level = 2**(i-1)+1
                end_level = 2**i
            elif i == 1:
                start_level = 1
                end_level = 2
            else:
                start_level = 0
                end_level = 0
            level_range = range(start_level, end_level+1)

            basis_functions.extend(ChebyshevFirstKind(n) for n in level_range)
            levels.append(list(level_range))
            for p in basis_functions[end_level].points:
                if not numpy.isclose(points, p).any():
                    points.append(p)
        return NestedBasisFunctionSet(points,basis_functions,levels)


class BasisFunctionSet():
    """Set of basis functions and sample points.

    """

    def __init__(self,points,basis_functions):
        self._basis_functions = basis_functions
        self._points = points
        if len(basis_functions) != len(points):
            raise IndexError("basis_functions and points must have the "
                    "same number of elements.")

    @property
    def points(self):
        """list: Sampling points."""
        return self._points

    @property
    def basis_functions(self):
        """list: Basis functions."""
        return self._basis_functions


class NestedBasisFunctionSet(BasisFunctionSet):
    """Nested set of basis functions and points.

    Nested points/basis function grow in levels, such that an approximation
    of a given level uses not only its sampling points but also all the points at
    lower levels. Nested sets (similarly to ogres) are like onions.
    """

    def __init__(self,points,basis_functions,levels):
        super().__init__(points,basis_functions)
        self._levels = levels

    @property
    def levels(self):
        """list: List of lists of indexes for points/functions at each level."""
        return self._levels

    @levels.setter
    def levels(self,levels):
        self._levels = levels

