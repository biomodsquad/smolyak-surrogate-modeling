import abc
import numpy


class UnidimensionalPointSet(abc.ABC):
    """Set of unidimensional points

    A set of unique unidimensional points to be used as the sampling points
    in a tensor product grid or sparse grid. These unidimensional points are
    associated with certain polynomial families, and the combination of
    unique points and polynomial families are the basis for different types
    of quadrature rules that are used in numerical integration and in
    approximation.
    The natural domain of the points specified by :param:``domain`` is
    the domain in which all the possible points in the set lies within.
    ``points`` is the list of points that the UnidimensionalPointSet
    is expected to store.
    ``domain`` is the domain of the points generated by the point set.
    """

    def __init__(self):
        self._points = None
        self._valid_cache = True

    @property
    @abc.abstractmethod
    def domain(self):
        """list: Domain the sample points come from."""
        pass

    @property
    def points(self):
        """list: Points stored by the set."""
        if self._valid_cache:
            self._create()
        self._valid_cache = False
        return self._points

    @abc.abstractmethod
    def _create(self):
        r"""Generating the points

        An abstract method for generating the points stored by the point set.
        Called when the number of points requested is larger than the number
        of points available.

        Parameters
        ----------
        num_points : int
            The number of points to generate
        """
        pass


class TieredUnidimensionalPointSet(UnidimensionalPointSet):
    """Set of unidimensional points with levels

    A set of unique unidimensional points to be used as the sampling points
    in a tensor product grid or sparse grid. These unidimensional points are
    associated with certain polynomial families, and the combination of
    unique points and polynomial families are the basis for different types
    of quadrature rules that are used in numerical integration and in
    approximation.

    Parameters
    ----------
    max_level ; int
        the maximum level the points are used for
    """

    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
        self._levels = None

    @property
    def max_level(self):
        """int: maximum level to compute points for."""
        return self._max_level

    @max_level.setter
    def max_level(self, value):
        self._max_level = value
        self._valid_cache = True

    @property
    def levels(self):
        """list: level indices stored by the set."""
        if self._valid_cache:
            self._create()
        self._valid_cache = False
        return self._levels
    
    @abc.abstractmethod
    def _create(self):
        r"""Generating the points

        An abstract method for generating the points stored by the point set.
        Called when the number of points requested is larger than the number
        of points available.

        Parameters
        ----------
        num_points : int
            The number of points to generate

        Returns
        -------
        self
            The UnidimensionalPointSet
        """
        pass


class ClenshawCurtisPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Clenshaw Curtis sampling

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    Parameters
    ----------
    degree : int
        degree of the Chebyshev polynomial of the first kind to get extrema from
    """

    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([-1, 1])

    @property
    def degree(self):
        """int: degree of polynomial to create points for."""
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = value
        self._valid_cache = True

    def _create(self):
        r"""Generating the points

        Parameters
        ----------
        num_points : int
            The number of points to generate

        Returns
        -------
        self
            The UnidimensionalPointSet
        """
        if self.degree > 0:
            points = -numpy.cos(
                numpy.pi * numpy.linspace(0, self.degree, self.degree + 1) / self.degree
            )
        else:
            points = numpy.zeros(1)
        self._points = points


class NestedClenshawCurtisPointSet(TieredUnidimensionalPointSet):
    r"""Generate nested Clenshaw Curtis points in accordance with a growth rule

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    The nested Clenshaw Curtis points come from the nested extrema of the
    Chebyshev polynomials of the first kind :math:0, 2, 2^{k} where k is an
    integer.

    These extrema are symmetric about 0. When generating levels of points, each
    level with contain a certain number of symmetric pairs except the first which
    also includes 0. The number of points assigned to each level can vary.
    The default, exponential growth adds points such that the new extrema
    calculated from a Chebyshev polynomial 2^{k} is given its own level k-1.
    Alternative growth rules may delay the addition of new extrema and have
    intermediary empty levels, but the general order that points are added to
    levels remains the same.

    This class will generate the nested extrema in the order that corresponds
    to the exponential growth. For increasing k, the unique extrema of Chebyshev
    polynomial with degree 2^{k - 1} will be added followed by the extrema of the
    polynomial with degree 2^{k} , then 2^{k + 1} , then 2^{k + 2} and so on.

    Parameters
    ----------
    max_level ; int
        the maximum level the points are used for
    """

    def __init__(self, max_level):
        super().__init__(max_level)

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([-1, 1])

    def _create(self):
        r"""Generating the points

        Parameters
        ----------
        num_points : int
            The number of points to generate

        Returns
        -------
        self
            The UnidimensionalPointSet
        """
        points = [0]
        degree = 0
        counter = 0
        for i in range(1, self.max_level + 1):
            counter = counter + 1
            degree = 2**counter
            if counter == 1:
                indexes = numpy.linspace(0, degree, 2, dtype=int)
            else:
                indexes = numpy.linspace(1, degree - 1, degree - 1, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            new_points = list(-numpy.cos(numpy.pi * indexes / degree))
            points.extend(new_points)
        self._points = points


class TrigonometricPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0

    Parameters
    ----------
    frequency : int
        the frequency to take points from
    """

    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([0, 2 * numpy.pi])

    @property
    def frequency(self):
        """int: frequency to create points for."""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._valid_cache = True

    def _create(self):
        r"""Generating the points

        Parameters
        ----------
        num_points : int
            The number of points to generate

        Returns
        -------
        self
            The UnidimensionalPointSet
        """
        if self.frequency > 0:
            idx = numpy.linspace(1, self.frequency, self.frequency)
            points = (idx - 1) * 2 * numpy.pi / self.frequency
        else:
            points = numpy.zeros(1)
        self._points = points


class NestedTrigonometricPointSet(TieredUnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0
    """

    def __init__(self, max_level):
        super().__init__(max_level)

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([0, 2 * numpy.pi])

    def _create(self):
        r"""Generating the points

        Parameters
        ----------
        num_points : int
            The number of points to generate

        Returns
        -------
        self
            The UnidimensionalPointSet
        """
        points = []
        degree = 0
        counter = 0
        for i in range(self.max_level + 1):
            degree = 3**counter
            for idx in range(1, degree + 1):
                point = (idx - 1) * 2 * numpy.pi / degree
                if not numpy.isclose(points, point).any():
                    points.append(point)
            counter += 1
        self._points = points
