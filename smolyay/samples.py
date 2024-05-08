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
        self._max_level = None
        self._num_points = None
        self._start_level = None
        self._end_level = None

        self.max_level = max_level

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
        """numpy.ndarray: number of points per level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._num_points

    @property
    def start_level(self):
        """numpy.ndarray: the starting index of each level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._start_level

    @property
    def end_level(self):
        """numpy.ndarray: the ending index of each level."""
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
    r"""Generate nested Clenshaw Curtis points

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    The nested Clenshaw Curtis points come from the nested extrema of the
    Chebyshev polynomials of the first kind :math:`n` where n is part of a 
    sequence
     
    .. math::

    n = \begin{cases}
         0 & \text{ if } L = 0\\ 
         2^{L} & \text{ if } L > 0 
    \end{cases}

    where L is a whole number.

    As the extrema of the Chebyshev polynomials of degree :math:`n` are nested,
    meaning any :math:`n` will have extrema at the same points as every n that
    precedes it in the sequence. The extrema are organized into levels such that
    a level L will contain the extrema of Chebyshev polynomial n(L) that are
    not the extrema of any Chebyshev polynomial n(k) where k is a whole number
    and k < L.

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
        self._num_points = numpy.ones(self.max_level + 1, dtype=int)
        self._num_points[1:] = [rule(i) - rule((i-1)) for i in range(1, self.max_level + 1)]

        self._start_level = numpy.zeros_like(self._num_points)
        self._start_level[1:] = numpy.cumsum(self._num_points[:-1])
        
        self._end_level = self._start_level + self._num_points
        # points
        points = [0]
        degree = 0
        num_levels = self.max_level + 1
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
    r"""Set for Clenshaw Curtis slow exponential growth.

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    The nested Clenshaw Curtis points come from the nested extrema of the
    Chebyshev polynomials of the first kind :math:`n` where n is part of a 
    sequence
     
    .. math::

    n = \begin{cases}
         0 & \text{ if } k = 0\\ 
         2^{k} & \text{ if } k > 0 
    \end{cases}

    where k is a whole number. 

    As the extrema of the Chebyshev polynomials of degree :math:`n` are nested,
    meaning any :math:`n` will have extrema at the same points as every n that
    precedes it in the sequence. The extrema are organized into levels such that
    a level L that is nonempty will contain extrema of some degree n that are
    not found in any preceding n.

    The total number of unique points increases exponentially with increasing n.
    To limit the rate of increasing points with increasing level L to a linear
    rate :math:`2*L + 1`, the relationship between k and L is described by the
    following equation:

    ..math:

        k = \left \lceil \log_{2}(L) \right \rceil + 1

    As k does not always increase when L increases, the rate of new, unique 
    points remains linear with respect to L.

    """

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
