import abc
import math
import warnings

import numpy
import scipy.special

class BasisFunction(abc.ABC):
    """Basis function for interpolating data.

     A one-dimensional basis function is defined on the domain
     :math:`[-1,1]`. The function defines the :attr:`points` at
     which it should be sampled within this interval for interpolation.
     The function also has an associated :meth:`__call__` method
     for evaluating it at a point within its domain. Moreover,
     the first derivative of the function can be evaluated via
     :meth:`derivative`.
    """

    def __init__(self):
        self._points = []

    @property
    @abc.abstractmethod
    def points(self):
        """list: Sampling points for interpolation."""
        pass

    @abc.abstractmethod
    def __call__(self, x):
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

    @abc.abstractmethod
    def derivative(self, x):
        """Evaluate the first derivative of the basis function.

        Parameters
        ----------
        x : float
            one-dimensional point.

        Returns
        -------
        float
            Value of the derivative of the basis function.
        """
        pass


class ChebyshevFirstKind(BasisFunction):
    r"""Chebyshev polynomial of the first kind.

    The Chebyshev polynomial :math:`T_n` of degree *n* is defined by the
    recursive relationship:

    .. math::

        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

    The :attr:`points` for this polynomial are the extrema on the domain
    :math:`[-1,1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    Parameters
    ----------
    n : int
        Degree of the Chebyshev polynomial.
    """

    def __init__(self, n):
        super().__init__()
        self._n = n
        self._derivative_polynomial = None
        if n > 0:
            self._points = [-numpy.cos(numpy.pi*i/n) for i in range(n+1)]
        else:
            self._points = [0]

    @property
    def points(self):
        """list: Sampling points at extrema of polynomial."""
        return self._points

    @property
    def n(self):
        """int: Degree of polynomial."""
        return self._n

    def __call__(self, x):
        r"""Evaluate the basis function.

        The Chebyshev polynomial is evaluated using the combinatorial formula:

        .. math::

            T_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} {n \choose 2k} (x^2-1)^k x^{n-2k}

        for :math:`n \ge 2`, and by the direct formula for the other values of *n*.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[-1,1]`.

        Returns
        -------
        float
            Value of Chebyshev polynomial of the first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1,1]

        """
        if numpy.any([numpy.greater(x,1),numpy.less(x,-1)]):
            raise ValueError("Input is outside the domain [-1,1]")

        return scipy.special.eval_chebyt(self._n,x)

    def derivative(self, x):
        """Evaluate the derivative of ChebyshevFirstKind.

        The first derivative of Chebyshev polynomials of first kind is
        evaluated using the relation between Chebyshev polynomial of
        first kind and second kind.

        ..math::
            T_n'(x) = nU_{n-1}(x)

        Parameters
        ----------
        x: float
            input in [-1, 1] domain.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1,1].
        """
        if numpy.any([numpy.greater(x,1),numpy.less(x,-1)]):
            raise ValueError("Input is outside the domain [-1,1]")
        if self._derivative_polynomial is None:
            self._derivative_polynomial = ChebyshevSecondKind(self._n-1)
        return self.n*self._derivative_polynomial(x)

    @classmethod
    def make_nested_set(cls, exactness):
        """Create a nested set of Chebyshev polynomials.

        A nested set is created up to a given level of ``exactness``,
        which corresponds to a highest-order Chebyshev polynomial of
        degree ``n = 2**exactness``.

        Each nesting level corresponds to the increasing powers of 2 going up to
        ``2**exactness``, with the first level being a special case. The
        generating Chebyshev polynomials are hence of degree (0, 2, 4, 8, ...).
        Each new point added in a level is paired with a basis function of
        increasing order.

        For example, for an ``exactness`` of 3, the generating polynomials are
        of degree 0, 2, 4, and 8, at each of 4 levels. There are 1, 2, 2, and 4
        new points added at each level. The polynomial at level 0 is of degree
        0, the polynomials at level 1 are of degrees 1 and 2, those at level 2
        are of degree 3 and 4, and those at level 3 are of degrees 5, 6, 7, and
        8.

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
        return NestedBasisFunctionSet(points, basis_functions, levels)


class ChebyshevSecondKind(BasisFunction):
    r"""Chebyshev polynomial of the second kind.

    The Chebyshev polynomial :math:`U_n` of degree *n* is defined by the
    recursive relationship:

    .. math::

        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)

    The :attr:`points` for this polynomial are the zeros on the domain
    :math:`[-1,1]`:

    .. math::

        x_i^* = -\cos(\pi i/(n+1)), i = 1,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    Parameters
    ----------
    n : int
        Degree of the Chebyshev polynomial.
    """

    def __init__(self, n):
        super().__init__()
        self._n = n
        self._derivative_polynomial = None
        if n > 1:
            self._points = [-numpy.cos(k*numpy.pi/(n+1)) for k in range(1, n+1)]
        else:
            self._points = [0.]

    @property
    def points(self):
        """list: Sampling points at extrema of polynomial."""
        return self._points

    @property
    def n(self):
        """int: Degree of polynomial"""
        return self._n

    def __call__(self,x):
        r"""Evaluate the basis function.

        The Chebyshev polynomial is evaluated using the combinatorial formula:

        .. math::

            U_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} {n+1 \choose 2k+1} (x^2-1)^k x^{n-2k}

        for :math:`n \ge 2`, and by the direct formula for the other values of *n*.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[-1,1]`.

        Returns
        -------
        float
            Value of Chebyshev polynomial of the second kind.

        Raises
        -------
        float
            Value of Chebyshev polynomial of the second kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1,1]

        """
        if numpy.any([numpy.greater(x,1),numpy.less(x,-1)]):
            raise ValueError("Input is outside the domain [-1,1]")
        return scipy.special.eval_chebyu(self._n,x)

    def derivative(self, x):
        r"""Evaluate the derivative of Chebyshev Second Kind.

        The first derivative of Chebyshev polynomials of second kind is
        evaluated using the connection between Chebyshev polynomial of
        first kind and second kind.

        ..math::
            U_n'(x) = \frac{(n+1)T_{n+1}(x)-xU_n(x)}{x^{2}-1}

        The above equation does not converge for :math:x={-1, 1}.

        ..math::
            \lim_{x \to 1} U_n'(x) = \frac{n(n+1)(n+2)}{3}
            \lim_{x \to -1} U_n'(x) = (-1)^{n+1} \frac{n(n+1)(n+2)}{3}

        derivative can be found using L'Hôpital's rule, 

        ..math::
            \lim_{x \to 1} U_n'(x) = \frac{U_n'(x)((n+1)(n+1)-1)}{3x}
            \lim_{x \to -1} U_n'(x) = \frac{U_n'(x)((n+1)(n+1)-1)}{3x}

        Parameters
        ----------
        x: float
            input in [-1, 1] domain.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1,1].
        """
        if numpy.any([numpy.greater(x,1),numpy.less(x,-1)]):
            raise ValueError("Input is outside the domain [-1,1]")
        if self._derivative_polynomial is None:
            self._derivative_polynomial = ChebyshevFirstKind(self._n+1)
        x = numpy.array(x, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            y = numpy.where(numpy.logical_not(numpy.logical_or(x == 1,x == -1)),
                            ((self._n+1)*self._derivative_polynomial(x) -
                             x*self(x))/(x**2-1),
                            self(x)*((self._n+1)**2 -1)/(3*x))
        if y.shape == ():
            return y[()]
        else:
            return numpy.squeeze(y)

    @classmethod
    def make_nested_set(cls, exactness):
        """Create a nested set of Chebyshev polynomials.

        A nested set is created up to a given level of ``exactness``,
        which corresponds to a highest-order Chebyshev polynomial of
        degree ``n = 2**(exactness + 1) - 1``.

        Each nesting level corresponds to the increasing powers of 2 going up to
        ``2**(exactness + 1) - 1``, with the first level being a special case.
        The generating Chebyshev polynomials are hence of degree (1, 3, 7,
        15, ...). Each new point added in a level is paired with a basis
        function of increasing order.

        For example, for an ``exactness`` of 3, the generating polynomials are
        of degree 1, 3, 7, and 16, at each of 4 levels. There are 2, 2, 4, and 8
        new points added at each level. The polynomials at level 0 are of degree
        0 and 1, the polynomials at level 1 are of degrees 2 and 3, those at
        level 2 are of degree 4, 5, 6, and 7, and those at level 3 are of
        degrees 8, 9, 10, 11, 12, 13, 14, and 15.

        Parameters
        ----------
        exactness : int
            Level of exactness.

        Returns
        -------
        NestedBasisFunctionSet
            Nested Chebyshev polynomials of the first kind.

        """
        # initialize 0th level to ensure it has 2 points
        levels = [[0,1]]
        basis_functions =  [ChebyshevSecondKind(0),ChebyshevSecondKind(1)]
        points = basis_functions[0].points + basis_functions[1].points
        for i in range(1, exactness+1):
            start_level = 2**i
            end_level = 2**(i+1)-1
            level_range = range(start_level, end_level+1)

            basis_functions.extend(ChebyshevSecondKind(n) for n in level_range)
            levels.append(list(level_range))
            for p in basis_functions[end_level].points:
                if not numpy.isclose(points, p).any():
                    points.append(p)
        return NestedBasisFunctionSet(points,basis_functions,levels)


class BasisFunctionSet():
    """Set of basis functions and sample points.

    Parameters
    ----------
    basis_functions : list
        Basis functions in set.

    points : list
        Sample point corresponding to each basis function.

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
    of a given level uses not only its sampling points but also all the points
    at lower levels. Nested sets (similarly to ogres) are like onions.

    Parameters
    ---------
    levels : list of lists
       Assignment of points/basis functions to each level.
    """

    def __init__(self,points,basis_functions,levels):
        super().__init__(points,basis_functions)
        self._levels = levels

    @property
    def levels(self):
        """list: List of lists of indexes for points/functions at each level.
        Raises
        ------
        IndexError
            max index must be less than total number of points.
        """
        return self._levels

    @levels.setter
    def levels(self,levels):
        if numpy.any(numpy.concatenate(levels) > len(self.points)):
            raise IndexError("max level index must be less than total "
                    "number of points.")
        self._levels = levels

