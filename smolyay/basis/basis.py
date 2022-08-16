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
        """The points of the basis function assigned to Smolyak indices"""
        if len(self._points) == 0:
            self._compute_points()
        return self._points

    @abc.abstractmethod
    def _compute_points(self):
        """Compute the value of the points for the basis function
        Computes the 1D points associated with the basis function, The
        method of computation depends on the basis function and what
        the points are meant to represent
        """
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


class BasisFunctionSet(abc.ABC):
    """Set of basis functions
    Creates a framework for a set of basis functions to be used as the
    building blocks for a surrogate model.
    ``basis_set`` is a list of BasisFunction objects
    ``sample_flag`` is a list of boolean equal in length to basis_set and
    an index in the list is true if the object in basis_set at the same
    index is one that points should be sampled from
    ``basis_function`` is the BasisFunction child class that populates
    basis_set
    ''all_points'' 1D points taken from the BasisFunction objects specified
    by sample_flag
    ``need_update`` is a boolean parameter that specifies when properties
    of the class need to be recomputed

    """

    def __init__(self,sample_flag,basis_function):
        self._sample_flag = sample_flag
        self._basis_function = basis_function
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
        if self._need_update:
            self._update()
        return self._basis_set

    @property
    def sample_flag(self):
        """list of what objects in basis_set to sample points from"""
        return self._sample_flag

    @sample_flag.setter
    def sample_flag(self,sample_flag):
        self._sample_flag = sample_flag
        self._need_update = True

    @property
    def basis_function(self):
        """BasisFunction class used in basis_set"""
        return basis_function

    @basis_function.setter
    def basis_function(self,basis_function):
        self._basis_function = basis_function
        self._need_update = True

    @abc.abstractmethod
    def _update(self):
        """Updates all_points based on parameters"""
        pass


class ChebyshevSet(BasisFunctionSet):
    """Set of the family of Chebyshev polynomials
    Set of Chebyshev polynomials to form the surrogate function

    The set of basis functions this class uses are assumed to be part
    of a sequential series of functions, where the index of a function
    in basis_set is the only required input of the basis_function
    """

    def __init__(self,sample_flag,basis_function):
        """Initialization of parameters"""
        super().__init__(sample_flag,basis_function)
        self._update()

    def _update(self):
        """Update basis_set and get points"""
        self._basis_set = []
        self._all_points = []

        for i in range(0,len(self._sample_flag)):
            self._basis_set.append(self._basis_function(i))

        for i in range(0,len(self._sample_flag)):
            if self._sample_flag[i]:
                new_points = self._basis_set[i].points
                for j in range(0,len(new_points)):
                    if new_points[j] not in self._all_points:
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
