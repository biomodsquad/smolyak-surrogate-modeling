import abc
import sys
import numpy


class UnidimensionalPointSet(abc.ABC):
    """Set of unidimensional points

    A set of unique unidimensional points to be used as the sampling points
    in a tensor product grid or sparse grid. These unidimensional points are
    associated with certain polynomial families, and the combination of
    unique points and polynomial families are the basis for different types
    of quadrature rules that are used in numerical integration and in
    approximation.
    The natural domain of the points specified by :param:``natural_domain`` is
    the domain in which all the possible points in the set lies within.
    ``number_points`` is the number of points that the UnidimensionalPointSet
    is expected to store.

    Parameters
    ----------
    natural_domain : list of two numbers
        the natural domain of the function within which calls are made.
    """

    def __init__(self):
        self._points = []
        self._number_points = 0

    @property
    @abc.abstractmethod
    def domain(self):
        """list: Domain the sample points come from."""
        pass

    @property
    def points(self):
        """list: Points stored by the set."""
        return self._points

    @property
    def number_points(self):
        """int: The number of points stored by the set."""
        return self._number_points

    @number_points.setter
    def number_points(self, value):
        if not isinstance(value, int):
            raise TypeError("Number must be integer greater than 0.")
        if value <= 0:
            raise ValueError("Number of points must be greater than 0")
        if len(self._points) < value:
            self._points = self._generate_points(value)
        self._points = self._points[:value]
        self._number_points = len(self._points)

    @abc.abstractmethod
    def _generate_points(self, num_points):
        r"""Generating the points

        An abstract method for generating the points stored by the point set.
        Called when the number of points requested is larger than the number
        of points available.

        Parameters
        ----------
        x : int
            The number of points to generate

        Returns
        -------
        list
            points stored by the set
        """
        pass


class ClenshawCurtisPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Clenshaw Curtis sampling

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.
    """

    def __init__(self):
        super().__init__()

    @property
    def domain(self):
        """list: Domain the sample points come from."""
        return [-1, 1]

    def _generate_points(self, num_points):
        r"""Generating the points

        Parameters
        ----------
        x : int
            The number of points to generate

        Returns
        -------
        list
            Value of Chebyshev polynomial of the first kind.
        """
        points = [0]
        degree = 0
        counter = 0
        while len(points) < num_points:
            counter = counter + 1
            degree = 2**counter
            if counter == 1:
                indexes = numpy.linspace(0, degree, 2, dtype=int)
            else:
                indexes = numpy.linspace(1, degree - 1, degree - 1, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            new_points = list(-numpy.cos(numpy.pi * indexes / degree))
            points.extend(new_points)
        return points


class TrigonometricPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0
    """

    def __init__(self):
        super().__init__()

    @property
    def domain(self):
        """list: Domain the sample points come from."""
        return [0, 2 * numpy.pi]

    def _generate_points(self, num_points):
        r"""Generating the points

        Parameters
        ----------
        x : int
            The number of points to generate

        Returns
        -------
        list
            Value of Chebyshev polynomial of the first kind.
        """
        points = []
        degree = 0
        counter = 0
        while len(points) < num_points:
            degree = 3**counter
            for idx in range(1, degree + 1):
                point = (idx - 1) * 2 * numpy.pi / degree
                if not numpy.isclose(points, point).any():
                    points.append(point)
            counter += 1
        return points
