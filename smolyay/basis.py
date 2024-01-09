import abc
import math

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
            self._points = -numpy.cos(numpy.pi*numpy.linspace(0,n,n+1)/n)
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
        if numpy.any(numpy.greater(x, 1)) or numpy.any(numpy.less(x, -1)):
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
        if numpy.any(numpy.greater(x, 1)) or numpy.any(numpy.less(x, -1)):
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
                if not numpy.isclose(points, p,rtol=0,atol=1e-11).any():
                    points.append(p)
        return NestedBasisFunctionSet(points, basis_functions, levels)

    @classmethod
    def make_slow_nested_set(cls, exactness,rule = lambda x : 2*x + 1):
        r"""Create a slow growth nested set of Chebyshev polynomials.

        A nested set is created up to a given level of ``exactness``,
        which corresponds to a highest-order Chebyshev polynomial of
        degree ``n = 2**exactness``.
        
        Each nesting level corresponds to the increasing powers of 2 going up to
        ``2**exactness``, with the first level being a special case. The 
        generating Chebyshev polynomials are hence of degree (0, 2, 4, ...).
        Each new point added in a level is paired with a basis function of 
        increasing order.
        
        The total number of 1D points accumulated at a given level i is given by
        the following equation.

        ..math::

            points(i) = \left\{ \begin{array}{cl} 1 & : \ i = 0 \\ 
                2^{i} + 1 & : \ otherwise \end{array} \right.

        The precision rule is a function that grows slower than the point
        accumulation, where the default is

        ..math::

            precision(exactness) = 2* ``exactness`` + 1

        For a given ``exactness``, points from the next generating Chebyshev
        polynomials are added at levels where the precision rule at the 
        current exactness is greater than the number of points accumulated at 
        previous levels.
        
        For example, for an ``exactness`` of 3, the generating polynomials are
        of degree 0, 2, 4, and 8, at each of 4 levels. There are 1, 2, 2, and 4
        new points added at each level, and the number of points accumulated
        at each level is 1, 3, 5, and 9. The polynomial at level 0 is of degree 
        0, the polynomials at level 1 are of degrees 1 and 2, those at level 2 
        are of degree 3 and 4, and those at level 3 are of degrees 5, 6, 7, and
        8.
        
        For an ``exactness`` of 4, there would be 1, 2, 2, 4, and 8 new points
        added at each level. However, the precision rule states that for level
        4 the precision rule is 9, and the accumulated number of points
        already added is 9. Since the precision rule is not greater than the 
        accumulated number of points, no new points are added. If the 
        ``exactness increases to 5, the precision rule will be greater than 9,
        and the 8 points that were skipped at ``exactness`` 4 are added.
    
        Parameters
        ----------
        exactness : int
            Level of exactness.

        rule : func
            custom precision rule.
            
        Returns
        -------
        NestedBasisFunctionSet
            Nested Chebyshev polynomials of the first kind.
        """
        basis_functions = []
        levels = []
        points = []
        rule_add = -1 # tracks rules added
        precision_has = 0 # tracks precision variable
        for i in range(0, exactness+1):
            if rule(i) > precision_has:
                rule_add = rule_add + 1
                if rule_add > 1:
                    start_level = 2**(rule_add-1)+1
                    end_level = 2**rule_add
                elif rule_add == 1:
                    start_level = 1
                    end_level = 2
                else:
                    start_level = 0
                    end_level = 0
                level_range = range(start_level, end_level+1)
                precision_has = precision_has + len(level_range)

                basis_functions.extend(ChebyshevFirstKind(n) for n in level_range)
                levels.append(list(level_range))
                for p in basis_functions[end_level].points:
                    if not numpy.isclose(points, p,rtol=0,atol=1e-11).any():
                        points.append(p)
            else:
                levels.append([])
        return NestedBasisFunctionSet(points,basis_functions,levels)


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
            self._points = -numpy.cos(numpy.pi*numpy.linspace(1,n,n)/(n+1))
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
        if numpy.any(numpy.greater(x, 1)) or numpy.any(numpy.less(x, -1)):
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
        Derivative can be found using L'Hôpital's rule.
        ..math::
            \lim_{x \to 1} U_n'(x) = \frac{n(n+1)(n+2)}{3}
            \lim_{x \to -1} U_n'(x) = (-1)^{n+1} \frac{n(n+1)(n+2)}{3}

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
        if numpy.any(numpy.greater(x, 1)) or numpy.any(numpy.less(x, -1)):
            raise ValueError("Input is outside the domain [-1,1]")
        if self._derivative_polynomial is None:
            self._derivative_polynomial = ChebyshevFirstKind(self._n+1)
        x = numpy.asarray(x)
        y = numpy.zeros(x.shape)
        flag2 = x == 1
        flag3 = x == -1
        flag1 = ~(flag2 | flag3)
        y[flag1] =  ((self._n+1)*self._derivative_polynomial(x[flag1]) -
                     x[flag1]*self(x[flag1]))/(x[flag1]**2-1)
        y[flag2] = self._n*(self._n + 1)*(self._n + 2)/3
        y[flag3] = ((-1)**(self._n+1))*self._n*(self._n + 1)*(self._n + 2)/3
        if y.ndim == 0:
            y = y.item()
        return y

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
                if not numpy.isclose(points, p,rtol=0,atol=1e-11).any():
                    points.append(p)
        return NestedBasisFunctionSet(points,basis_functions,levels)

    @classmethod
    def make_slow_nested_set(cls, exactness, rule= lambda x : 2*x + 1):
        r"""Create a nested set of Chebyshev polynomials of the second kind.

        A nested set is created up to a given level of ``exactness``,
        which corresponds to a highest-order Chebyshev polynomial of
        degree ``n = 2**(exactness + 1) - 1``.

        Each nesting level corresponds to the increasing powers of 2 going up to
        ``2**(exactness + 1) - 1``, with the first level being a special case.
        The generating Chebyshev polynomials are hence of degree (1, 3, 7,
        15, ...). Each new point added in a level is paired with a basis
        function of increasing order.

        The total number of 1D points accumulated at a given level i is given by
        the following equation.

        ..math::

            points(i) = 2^{i+1}

        The precision rule is a function that grows slower than the point
        accumulation, where the default is

        ..math::

            precision(exactness) = 2* ``exactness`` + 1

        For a given ``exactness``, points from the next generating Chebyshev
        polynomials are added at levels where the precision rule at the 
        current exactness is greater than the number of points accumulated at 
        previous levels.
        
        For example, for an ``exactness`` of 2, the generating polynomials are
        of degree 1, 3, and 7 at each of 3 levels. There are 2, 2, and 4
        new points added at each level, and the number of points accumulated
        at each level is 2, 4, and 8. The polynomial at level 0 is of degree 
        0 and 1, the polynomials at level 1 are of degrees 2 and 3, and those at 
        level 2 are of degree 4 and 5.
        
        For an ``exactness`` of 3, there would be 2, 2, 4, and 8 new points
        added at each level. However, the precision rule states that for level
        3 the precision rule is 7, and the accumulated number of points
        already added is 8. Since the precision rule is not greater than the 
        accumulated number of points, no new points are added. If the 
        ``exactness increases to 5, the precision rule will be greater than 8,
        and the 8 points that were skipped at ``exactness`` 3 are added.

        Parameters
        ----------
        exactness : int
            Level of exactness.

        rule : func
            custom precision rule.

        Returns
        -------
        NestedBasisFunctionSet
            Nested Chebyshev polynomials of the first kind.

        """
        # initialize 0th level to ensure it has 2 points
        levels = [[0,1]]
        basis_functions =  [ChebyshevSecondKind(0),ChebyshevSecondKind(1)]
        points = basis_functions[0].points + basis_functions[1].points
        rule_add = 0
        precision_has = 2
        for i in range(1, exactness+1):
            if rule(i) > precision_has:
                rule_add += 1
                start_level = 2**rule_add
                end_level = 2**(rule_add+1)-1
                level_range = range(start_level, end_level+1)
                precision_has += len(level_range)

                basis_functions.extend(ChebyshevSecondKind(n) for n in level_range)
                levels.append(list(level_range))
                for p in basis_functions[end_level].points:
                    if not numpy.isclose(points, p,rtol=0,atol=1e-11).any():
                        points.append(p)
            else:
                levels.append([])
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

