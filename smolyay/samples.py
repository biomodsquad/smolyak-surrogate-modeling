import abc
import numpy


class UnidimensionalPointSet(abc.ABC):
    """Set of unidimensional points

    A set of unique unidimensional points within a domain.
    """

    def __init__(self):
        self._points = None
        self._valid_cache = False

    @property
    @abc.abstractmethod
    def domain(self):
        """numpy.ndarray: Domain of the `points`."""
        pass

    @property
    def points(self):
        """numpy.ndarray: Points in the set."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._points

    @abc.abstractmethod
    def _create(self):
        """Create the points in the set."""
        pass


class NestedUnidimensionalPointSet(UnidimensionalPointSet):
    """Set of unidimensional points assigned to levels.


    Parameters
    ----------
    max_level ; int
        the maximum level the points are used for
    """

    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
        self._num_points = None
        self._start_level = None
        self._end_level = None

    @property
    def max_level(self):
        """int: maximum level to compute points for."""
        return self._max_level

    @max_level.setter
    def max_level(self, value):
        self._max_level = value
        self._valid_cache = False

    @property
    def num_points(self):
        """list: number of points per level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._num_points

    @property
    def start_level(self):
        """list: the starting index of each level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._start_level

    @property
    def end_level(self):
        """list: the ending index of each level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._end_level


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
        self._valid_cache = False

    def _create(self):
        r"""Create the points in the set.

        Generating the extrema of a Cheybshev polynomial of the first kind at
        a given degree.
        """
        if self.degree > 0:
            points = -numpy.cos(
                numpy.pi * numpy.linspace(0, self.degree, self.degree + 1) / self.degree
            )
        else:
            points = numpy.zeros(1)
        self._points = points


class NestedClenshawCurtisPointSet(NestedUnidimensionalPointSet):
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
        """numpy.ndarray: Domain of the `points`."""
        return numpy.array([-1, 1])

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels
        rule = lambda x: 1 if x == 0 else 2**x + 1
        self._num_points = [rule(0)] + [
            rule(i) - rule(i - 1) for i in range(1, self.max_level + 1)
        ]
        self._start_level = numpy.cumsum([0] + list(self._num_points))[:-1]
        self._end_level = numpy.cumsum(list(self._num_points))
        # points
        points = [0]
        degree = 0
        num_levels = numpy.sum(numpy.array(self._num_points) != 0)
        for i in range(1, num_levels):
            degree = 2**i
            if i == 1:
                indexes = numpy.linspace(0, degree, 2, dtype=int)
            else:
                indexes = numpy.linspace(1, degree - 1, degree - 1, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            new_points = list(-numpy.cos(numpy.pi * indexes / degree))
            points.extend(new_points)
        self._points = points


class SlowNestedClenshawCurtisPointSet(NestedClenshawCurtisPointSet):
    """Set for Clenshaw Curtis slow exponential growth"""

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels
        rule = lambda x: 1 if x == 0 else int(2 ** (numpy.ceil(numpy.log2(x)) + 1) + 1)
        self._num_points = [rule(0)] + [
            rule(i) - rule(i - 1) for i in range(1, self.max_level + 1)
        ]
        self._start_level = numpy.cumsum([0] + list(self._num_points))[:-1]
        self._end_level = numpy.cumsum(list(self._num_points))
        # points
        points = [0]
        degree = 0
        num_levels = numpy.sum(numpy.array(self._num_points) != 0)
        for i in range(1, num_levels):
            degree = 2**i
            if i == 1:
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
        """numpy.ndarray: Domain of the `points`."""
        return numpy.array([0, 2 * numpy.pi])

    @property
    def frequency(self):
        """int: frequency to create points for."""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._valid_cache = False

    def _create(self):
        r"""Create the points in the set.

        Generating trigonometic points at a given frequency.
        """
        if self.frequency > 0:
            idx = numpy.linspace(1, self.frequency, self.frequency)
            points = (idx - 1) * 2 * numpy.pi / self.frequency
        else:
            points = numpy.zeros(1)
        self._points = points


class NestedTrigonometricPointSet(NestedUnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0

    These points are nested, such that the order of elements in
    `points` corresponds to the indices in `levels`.
    """

    def __init__(self, max_level):
        super().__init__(max_level)

    @property
    def domain(self):
        """numpy.ndarray: Domain of the `points`."""
        return numpy.array([0, 2 * numpy.pi])

    def _create(self):
        r"""Create the points in the set.

        Generating the trignometric points using the frequencies
        :math:1, 3, 9, ..., 3^{i} where i is an integer.
        """
        # create properties for levels
        rule = lambda x: 3**x
        self._num_points = [rule(0)] + [
            rule(i) - rule(i - 1) for i in range(1, self.max_level + 1)
        ]
        self._start_level = numpy.cumsum([0] + list(self._num_points))[:-1]
        self._end_level = numpy.cumsum(list(self._num_points))
        # points
        points = []
        degree = 0
        for i in range(self.max_level + 1):
            degree = 3**i
            for idx in range(1, degree + 1):
                point = (idx - 1) * 2 * numpy.pi / degree
                if not numpy.isclose(points, point).any():
                    points.append(point)
        self._points = points