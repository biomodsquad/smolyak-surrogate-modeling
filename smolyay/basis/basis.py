import abc
import math

import numpy

class BasisFunction(abc.ABC):
    """Abstract class for basis functions.

    Creates a framework for basis functions that are the building blocks
    for the terms in the surrogate model. 
    ``points`` property represents all the unique 1D points of the 
    basis function associated with a Smolyak index described by the 
    class IndexGrid. The Smolyak index of a given point is equal to 
    its position in the list points.
    ``levels`` property gives the Smolyak indexes for each
    grid level.

    """

    def __init__(self):
        self._points = []
        self._levels = []

    @property
    def points(self):
        """Extrema assigned to Smolyak indices"""
        return self._points

    @property
    def levels(self):
        """index of each point that belongs to each grid level"""
        return self._levels

    @abc.abstractmethod
    def __call__(self,n,x):
        """Compute term of basis function
        Returns the output of an nth degree basis function with input x

        Parameters
        ----------
        n : int
            degree

        x : float
            input

        Returns
        -------
        y : float
            output
        """
        pass


class ChebyshevFirstKind(BasisFunction):
    r"""Basis function of Chebyshev polynomials of the first kind
    Chebyshev polynomials are a sequence of polynomials described by the
    following recurrence relation:

    ..math:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)

    ``max_exactness`` represents the maximum level of exactness, or
    accuracy, of a surrogate function this class can provide enough
    information for. It determines the number of points and levels
    the class will calculate and store.

    The extrema of the polynomials are calculated via the following
    equation:

    ..math:
        x_{n,j}^* = -\cos\left((\frac{j-1}{n-1}\pi\right), j = 1,...,n
        n = 2^{exactness-1}+1

    Parameters
    ----------
    max_exactness : int
        level of exactness class will describe
    """

    def __init__(self,max_exactness):
        super().__init__()
        self._max_exactness = max_exactness
        self._update()

    @property
    def max_exactness(self):
        """Maximum level of exactness described"""
        return self._max_exactness

    @max_exactness.setter
    def max_exactness(self, max_exactness):
        """Set maximum level of exactness
        Sets maximmum level of exactness class instance will describe, will
        extend or truncate properties points and levels to match new value

        Parameters
        ----------
        max_exactness : int
            new level of exactness class should describe
        """
        if max_exactness > self._max_exactness:
            # extend
            self._max_exactness = max_exactness
            # update other properties
            self._update()
        elif max_exactness < self._max_exactness:
            # truncate
            self._max_exactness = max_exactness
            # update other properties
            points_keep = sum(
                    [len(x) for x in self._levels[:self._max_exactness+1]])
            self._points = self._points[:points_keep]
            self._levels = self._levels[:self._max_exactness+1]

    def _update(self):
        """Update properties dependent on max_exactness"""
        next_exactness = len(self._levels)

        if next_exactness == 0:
            self._points = [0]
            self._levels = [[0]]
            next_exactness = 1

        for ex in range(next_exactness,self._max_exactness+1):
            n = 2**(ex)+1
            counter_index = 0
            new_level = []
            for i in range(1,n+1):
                temp = round(-numpy.cos(numpy.pi*(i-1)/(n-1)),15) + 0
                if temp not in self._points:
                    new_level.append(len(self._points))
                    self._points.append(temp)
                    counter_index += 1
            self._levels.append(new_level)

    def __call__(self,n,x):
        """Terms of basis function
        Returns the output of an nth degree Chebyshev polynomial of the
        first kind with input x

        Parameters
        ----------
        n : int
            degree of Chebyshev polynomial of the first kind

        x : float
            input

        Returns
        -------
        y : float
            output
        """
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            k_lim = n//2
            answer = 0
            for k in range(0,k_lim+1):
                answer += math.comb(n,2*k)*((x**2 - 1)**k)*(x**(n-2*k))
            return answer


