import abc
import math

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

    @property
    def points(self):
        """Extrema assigned to Smolyak indices"""
        return self._points

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

class BasisFunctionSet():
    """Set of basis functions

    Creates a framework for a set of basis functions to be used as the
    building blocks for a surrogate model.
    ``basis_set'' is a list of BasisFunction objects
    ``sample_flag`` is a list of boolean equal in length to basis_set and
    an index in the list is true if the object in basis_set at the same
    index is one that points should be sampled from
    ''all_points'' 1D points taken from the BasisFunction objects specified
    by sample_flag
    """

    def __init__(self,sample_flag):
        self._sample_flag = sample_flag
        self._basis_set = []
        self._all_points = []
        self._need_update = True

    @property
    def all_points(self):
        """All points for basis function set at some level of precision"""
        if self._need_update:
            self._update()
        return self._all_points

    @property
    def basis_set(self):
        """list of BasisFunction objects"""
        return self._basis_set

    @property
    def sample_flag(self):
        """list of what objects in basis_set to sample points from"""
        return self._sample_flag

    @sample_flag.setter(self,sample_flag):
        self._sample_flag = sample_flag
        self.need_update = True



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
            self._update()
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

    def _update(self):
        """Update properties dependent on max_exactness"""

        if self._n == 0:
            self._points = [0]
        else:
            for i in range(0,self._n+1):
                temp = round(-math.cos(math.pi*(i)/(self._n)),15) + 0
                self._points.append(temp)

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


