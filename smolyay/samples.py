import abc
import numpy


class UnidimensionalPointSet(abc.ABC):
    """Set of unidimensional points

    A set of unique unidimensional points within a domain.
    """

    def __init__(self, domain):
        self._points = None
        self._domain = numpy.array(domain)
        self._valid_cache = False

    @property
    def domain(self):
        """numpy.ndarray: Domain of the `points`."""
        return self._domain

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

    @staticmethod
    def scale_to_domain(x, old, new):
        print(x)
        new_x = new[0] + (new[1] - new[0]) * ((x - old[0]) / (old[1] - old[0]))
        numpy.clip(new_x, new[0], new[1], out=new_x)
        print(new_x)
        return new_x
    


class NestedUnidimensionalPointSet(UnidimensionalPointSet):
    """Set of unidimensional points assigned to levels.


    Parameters
    ----------
    max_level ; int
        the maximum level the points are used for
    """

    def __init__(self, domain, max_level):
        super().__init__(domain)
        self._max_level = None
        self._num_points_per_level = None
        self._start_level = None
        self._end_level = None

        self.max_level = max_level

    @property
    def max_level(self):
        """int: maximum level to compute points for."""
        return self._max_level

    @max_level.setter
    def max_level(self, value):
        max_level = int(value)
        if max_level != self._max_level:
            self._max_level = max_level
            self._valid_cache = False

    @property
    def num_points_per_level(self):
        """numpy.ndarray: number of points per level."""
        if not self._valid_cache:
            self._create()
        self._valid_cache = True
        return self._num_points_per_level

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

    def __init__(self, domain, degree):
        super().__init__(domain)
        self._degree = None

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
        degree = int(value)
        if degree != self._degree:
            self._degree = degree
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
        self._points = self.scale_to_domain(points, [-1, 1], self.domain)


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

    def __init__(self, domain, max_level):
        super().__init__(domain, max_level)

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels
        rule = lambda x: 1 if x == 0 else 2**x + 1
        self._num_points_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_points_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]

        self._start_level = numpy.zeros_like(self._num_points_per_level)
        self._start_level[1:] = numpy.cumsum(self._num_points_per_level[:-1])

        self._end_level = self._start_level + self._num_points_per_level
        # points
        points = numpy.zeros(numpy.sum(self._num_points_per_level))
        degree = 0
        num_levels = self.max_level + 1
        for i in range(1, num_levels):
            degree = 2**i
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            if i == 1:
                indexes = numpy.arange(0, degree + 1, 2, dtype=int)
            else:
                indexes = indexes = numpy.arange(1, degree, 2, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            # generate extrema
            new_points = list(-numpy.cos(numpy.pi * indexes / degree))
            points[self._start_level[i] : self._end_level[i]] = new_points
        self._points = self.scale_to_domain(points, [-1, 1], self.domain)


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
        self._num_points_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_points_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]

        self._start_level = numpy.zeros_like(self._num_points_per_level)
        self._start_level[1:] = numpy.cumsum(self._num_points_per_level[:-1])

        self._end_level = self._start_level + self._num_points_per_level
        # points
        points = numpy.zeros(numpy.sum(self._num_points_per_level))
        degree = 0
        num_levels = numpy.sum(numpy.array(self._num_points_per_level) != 0)
        nonempty_index = lambda x: 0 if x == 0 else int(2 ** (x - 2)) + 1
        for i in range(1, num_levels):
            degree = 2**i
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            if i == 1:
                indexes = numpy.arange(0, degree + 1, 2, dtype=int)
            else:
                indexes = indexes = numpy.arange(1, degree, 2, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            # generate extrema
            new_points = list(-numpy.cos(numpy.pi * indexes / degree))

            points[
                self._start_level[nonempty_index(i)] : self._end_level[
                    nonempty_index(i)
                ]
            ] = new_points
        self._points = self.scale_to_domain(points, [-1, 1], self.domain)


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

    def __init__(self, domain, frequency):
        super().__init__(domain)
        self._frequency = None

        self.frequency = frequency

    @property
    def frequency(self):
        """int: frequency to create points for."""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        frequency = int(value)
        if frequency != self._frequency:
            self._frequency = frequency
            self._valid_cache = False

    def _create(self):
        r"""Create the points in the set.

        Generating trigonometic points at a given frequency.
        """
        if self.frequency > 0:
            idx = numpy.arange(1, self.frequency + 1, 1, dtype=int)
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

    def __init__(self, domain, max_level):
        super().__init__(domain, max_level)

    def _create(self):
        r"""Create the points in the set.

        Generating the trignometric points using the frequencies
        :math:1, 3, 9, ..., 3^{i} where i is an integer.
        """
        # create properties for levels
        rule = lambda x: 3**x
        self._num_points_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_points_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]

        self._start_level = numpy.zeros_like(self._num_points_per_level)
        self._start_level[1:] = numpy.cumsum(self._num_points_per_level[:-1])

        self._end_level = self._start_level + self._num_points_per_level
        # points
        points = []
        degree = 0
        for i in range(self.max_level + 1):
            degree = 3**i
            for idx in range(1, degree + 1):
                point = (idx - 1) * 2 * numpy.pi / degree
                if not numpy.isclose(points, point).any():
                    points.append(point)
        points = numpy.array(points)
        self._points = self.scale_to_domain(points, [0, 2 * numpy.pi], self.domain)
