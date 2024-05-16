import abc
import numpy


class UnidimensionalPointSet(abc.ABC):
    """Set of unidimensional points

    A set of unique unidimensional points within a domain.

    Parameters
    ----------
    domain: list
        Domain of the sample points.
    """

    def __init__(self, domain):
        self._points = None
        self._domain = None
        self._valid_cache = False

        self.domain = domain

    @property
    def domain(self):
        """numpy.ndarray: Domain of the `points`."""
        return self._domain

    @domain.setter
    def domain(self, value):
        domain = numpy.array(value, dtype=float)
        if domain.shape != (2,):
            raise TypeError("Domain must be array with two variables")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            self._valid_cache = False

    @property
    def points(self):
        """numpy.ndarray: Points in the set."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._points

    def __len__(self):
        return len(self.points)

    @abc.abstractmethod
    def _create(self):
        """Create the points in the set."""
        pass

    def _scale_to_domain(self, points, old_domain):
        points = self.domain[0] + (self.domain[1] - self.domain[0]) * (
            (points - old_domain[0]) / (old_domain[1] - old_domain[0])
        )
        numpy.clip(points, self.domain[0], self.domain[1], out=points)
        return points


class NestedUnidimensionalPointSet(UnidimensionalPointSet):
    """Set of unidimensional points assigned to levels.


    Parameters
    ----------
    domain: list
        Domain of the sample points.

    max_level ; int
        The maximum level the points are used for.
    """

    def __init__(self, domain, max_level):
        super().__init__(domain)
        self._max_level = None
        self._num_per_level = None
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
    def num_per_level(self):
        """numpy.ndarray: number of points per level."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._num_per_level

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
    domain: list
        Domain of the sample points.

    degree : int
        Degree of the Chebyshev polynomial of the first kind to get extrema from.
    """

    def __init__(self, domain, degree):
        super().__init__(domain)
        self._degree = None

        self.degree = degree

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
            # special case where degree == 0
            points = numpy.zeros(1)
        self._points = self._scale_to_domain(points, [-1, 1])


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

    The order :math:`o(L) of the nested Clenshaw Curtis set, which describes the
    number of cummulative points at each level, is used to determine the 
    number of points at each individual level. The order of the Clenshaw Curtis
    set is described by the following rule:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{L} + 1 & \text{ if } L > 0 
        \end{cases}

    which leads to a sequence :math:`{1, 3, 5, 9, 17, ...}`.

    Determining the number of points each level is then

    .. math::
        num_per_level(L) = o(L) - o(L - 1)

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    max_level: int
        The maximum level the points are used for.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels, level 0 is a special case with 1 point
        rule = lambda x: 1 if x == 0 else 2**x + 1
        self._num_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level

        # points, level 0 is a special case only 0 as a point
        num_points = self._end_level[-1]
        points = numpy.zeros(num_points, dtype=float)
        for i in range(1, self.max_level + 1):
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            degree = 2**i
            if i == 1:
                # special case for level == 1
                indexes = numpy.arange(0, degree + 1, 2, dtype=int)
            else:
                indexes = numpy.arange(1, degree, 2, dtype=int)
                divisible_indexes = numpy.gcd(indexes, degree) > 1
                indexes = indexes[~divisible_indexes]
            points[self._start_level[i] : self._end_level[i]] = -numpy.cos(
                numpy.pi * indexes / degree
            )
        self._points = self._scale_to_domain(points, [-1, 1])


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
    not found in any preceding n. The total number of unique points increases 
    exponentially with increasing n.
    
    To limit the rate of increasing points with increasing level L to a linear
    rate :math:`2*L + 1`, the relationship between k and L is described by the
    following equation:

    ..math:

        k = \left \lceil \log_{2}(L) \right \rceil + 1

    As k does not always increase when L increases, the rate of new, unique 
    points is capped at a rate less than or equal to :math:`2*L + 1`

    The order :math:`o(L) of the slow nested Clenshaw Curtis set, which 
    describes the number of cummulative points at each level, is used to 
    determine the number of points at each individual level. The order of the 
    Clenshaw Curtis set is described by the following rule:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{k} + 1 & \text{ if } L > 0 
        \end{cases}

        where k = \left \lceil \log_{2}(L) \right \rceil + 1

    The sequence of o(L) is then :math:`{1, 3, 5, 9, 9, 17, ...}`.
    
    Since the points are nested, the number of unique points at
    a level L is 
    
    .. math::
        num_per_level(L) = o(L) - o(L - 1)    

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    max_level: int
        The maximum level the points are used for.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels, level 0 is a special case with 1 point
        rule = lambda x: 1 if x == 0 else int(2 ** (numpy.ceil(numpy.log2(x)) + 1) + 1)
        self._num_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level

        # points, level 0 is a special case only 0 as a point
        points = numpy.zeros(numpy.sum(self._num_per_level))
        degree = 0
        for i in range(1, self.max_level + 1):
            if self._num_per_level[i] == 0:
                continue
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            degree = int(2**(numpy.ceil(numpy.log2(i)) + 1))
            if i == 1:
                # special case for level == 1
                indexes = numpy.arange(0, degree + 1, 2, dtype=int)
            else:
                indexes = indexes = numpy.arange(1, degree, 2, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            points[self._start_level[i] : self._end_level[i]] = -numpy.cos(
                numpy.pi * indexes / degree
            )
        self._points = self._scale_to_domain(points, [-1, 1])


class TrigonometricPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    frequency : int
        The frequency to take points from.
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
            # special case where degree == 0
            points = numpy.zeros(1)
        self._points = self._scale_to_domain(points, [0, 2 * numpy.pi])


class NestedTrigonometricPointSet(NestedUnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0

    These points are nested, such that the order of elements in
    `points` corresponds to the indices in `levels`.

    To determine the number of points per level, an order(L) is
    used to describe the number of points at each level L. For
    the Trigonometric points, this rule is

    .. math::

        o(L) = 3^{L}

    Since the points are nested, the number of unique points at
    a level L is

    .. math::
        num_per_level(L) = o(L) - o(L - 1)


    Parameters
    ----------
    domain: list
        Domain of the sample points.

    max_level: int
        The maximum level the points are used for.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating the trignometric points using the frequencies
        :math:1, 3, 9, ..., 3^{i} where i is an integer.
        """
        # create properties for levels, level 0 is a special case with 1 point
        rule = lambda x: 3**x
        self._num_per_level = numpy.ones(self.max_level + 1, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule((i - 1)) for i in range(1, self.max_level + 1)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level

        # points, level 0 is a special case only 0 as a point
        points = numpy.zeros(numpy.sum(self._num_per_level))
        for i in range(self.max_level + 1):
            # find fraction index/degree that cannot be further simplified
            degree = 3**i
            if i == 0:
                # special case where level == 0
                indexes = numpy.arange(0, degree, dtype=int)
            else:
                indexes = numpy.arange(1, degree + 1, 1, dtype=int)
                indexes = indexes[~(numpy.gcd(indexes, degree) > 1)]
            points[self._start_level[i] : self._end_level[i]] = (
                (indexes) * 2 * numpy.pi / degree
            )
        self._points = self._scale_to_domain(points, [0, 2 * numpy.pi])
