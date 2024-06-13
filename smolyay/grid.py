import abc
import itertools

import numpy
import scipy.stats.qmc

from smolyay.basis import BasisFunctionSet


class MultidimensionalPointSet(abc.ABC):
    """Multidimensional set of points

    Creates a set of grid points within the specified domain

    Parameters
    ----------
    domain: list
         the domain of the grid
    """

    def __init__(self):
        self._domain = None
        self._points = None
        self._valid_cache = False

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return numpy.shape(self.domain)[0]

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return self._domain

    @property
    def points(self):
        """numpy.ndarray: Points to be sampled."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._points

    def __len__(self):
        return self.points.shape[0]

    @abc.abstractmethod
    def _create(self):
        """Abstract method for generating the mulitdimensional grid set"""
        pass


class RandomPointSet(MultidimensionalPointSet):
    """Point Set that uses randomly generated points

    A Multidimensional point set that creates a grid using a Monte Carlo
    or Quasi Monte Carlo method for generating points.

    Parameters
    ----------
    domain: list
        domain of the random points

    number_points: int
        number of points to generate

    method: {"uniform", "halton", "sobol", "latin"}
        the method of generating random points

    seed: {int, numpy.random.Generator}
        seed for generating the random points

    options: {None, dict}, optional
        if method != None, additional parameters passed to the QMCEngine
    """

    def __init__(self, domain, number_points, seed):
        super().__init__()
        self._domain = None
        self._number_points = None
        self._seed = None

        self.domain = domain
        self.number_points = number_points
        self.seed = seed

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return self._domain

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise TypeError("Domain must have size (num_dimensions, 2)")
        if any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("Lower bound must be less than upper bound")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            self._valid_cache = False

    @property
    def number_points(self):
        """int: random seed for generating points"""
        return self._number_points

    @number_points.setter
    def number_points(self, value):
        number_points = int(value)
        if self._number_points != number_points:
            self._number_points = number_points
            self._valid_cache = False

    @property
    def seed(self):
        """int: random seed for generating points"""
        return self._seed

    @seed.setter
    def seed(self, value):
        if not isinstance(value, numpy.random.Generator):
            value = int(value)
        if self._seed != value:
            self._seed = value
            self._valid_cache = False

    def _create(self):
        """Generating the Monte Carlo mulitdimensional grid set

        Using the parameters of the class, generate a grid set using
        the a Monte Carlo/Quasi Monte Carlo method.
        """
        self._points = self._get_random_points()

    @abc.abstractmethod
    def _get_random_points(self):
        """Generate a set of random points

        Generates a set of random points with a given Monte Carlo/Quasi Monte
        Carlo method depending on the class.

        Returns
        -------
        numpy.ndarray
            generated random points
        """
        pass


class UniformRandomPointSet(RandomPointSet):
    """Generates a grid using a uniform distribution"""

    def _get_random_points(self):
        lower_bounds = [bound[0] for bound in self.domain]
        upper_bounds = [bound[1] for bound in self.domain]
        num_dimensions = len(lower_bounds)
        p_gen = numpy.random.default_rng(seed=self.seed).uniform(
            size=(self.number_points, num_dimensions)
        )
        return scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class QMCRandomPointSet(RandomPointSet):
    """Point set that generates points using a QMCEngine

    This random point set relies on the QMCEngine objects in
    the scipy.stats.qmc module to generate points.

    In addition to the domain, number of points, and the seed
    to generate the points, all QMCEngines have 2 additional
    optional parameters, ``scramble`` and ``optimization``.

    Parameters
    ----------
    domain: list
        domain of the random points

    number_points: int
        number of points to generate

    seed: {int, numpy.random.Generator}
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.
    """

    def __init__(self, domain, number_points, seed, scramble=True, optimization=None):
        super().__init__(domain, number_points, seed)
        self._scramble = True
        self._optimization = None

        self.scramble = scramble
        self.optimization = optimization

    @property
    def scramble(self):
        """bool: If True, applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points."""
        return self._scramble

    @scramble.setter
    def scramble(self, value):
        scramble = bool(value)
        if self._scramble != scramble:
            self._scramble = scramble
            self._scramble = False

    @property
    def optimization(self):
        """{None, “random-cd”, “lloyd”}: perform post-processing on points"""
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if not value in [None, "random-cd", "lloyd"]:
            raise TypeError("optimization must be None, random-cd, or lloyd.")
        if self._optimization != value:
            self._optimization = value
            self._valid_cache = False


class LatinHypercubeRandomPointSet(QMCRandomPointSet):
    """Generates a grid using a LatinHypercube


    Parameters
    ----------
    domain: list
        domain of the random points

    number_points: int
        number of points to generate

    seed: {int, numpy.random.Generator}
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    strength : {1, 2}, optional
        Default 1. If 1, produces a normal LHS. If 2, produces an orthogonal
        array based LHS.
    """

    def __init__(
        self, domain, number_points, seed, scramble=True, optimization=None, strength=1
    ):
        super().__init__(domain, number_points, seed, scramble, optimization)
        self._strength = 1

        self.strength = strength

    @property
    def strength(self):
        """{1, 2}: Whether to create an orthogonal array of points"""
        return self._strength

    @strength.setter
    def strength(self, value):
        strength = int(value)
        if not strength in [1, 2]:
            raise TypeError("Strength must be 1 or 2.")
        if self._strength != strength:
            self._strength = strength
            self._valid_cache = False

    def _get_random_points(self):
        lower_bounds = [bound[0] for bound in self.domain]
        upper_bounds = [bound[1] for bound in self.domain]
        p_gen = scipy.stats.qmc.LatinHypercube(
            self.num_dimensions,
            scramble=self.scramble,
            strength=self.strength,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.number_points)
        return scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class HaltonRandomPointSet(QMCRandomPointSet):
    """Generates a grid using Halton Sequences

    Parameters
    ----------
    domain: list
        domain of the random points

    number_points: int
        number of points to generate

    seed: {int, numpy.random.Generator}
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    bits : int
        Default 30. Sets the max number of points that can be generated,
        which is 2**bits.
    """

    def _get_random_points(self):
        lower_bounds = [bound[0] for bound in self.domain]
        upper_bounds = [bound[1] for bound in self.domain]
        p_gen = scipy.stats.qmc.Halton(
            self.num_dimensions,
            scramble=self.scramble,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.number_points)
        return scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class SobolRandomPointSet(QMCRandomPointSet):
    """Generates a grid using Sobol Sequence

    Parameters
    ----------
    domain: list
        domain of the random points

    number_points: int
        number of points to generate. Must be a power of 2.

    seed: {int, numpy.random.Generator}
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    Raises
    ------
    ValueError
        number of points must be a power of two
    ValueError 
        number of points must be less than 2**bits
    """

    def __init__(
        self, domain, number_points, seed, scramble=True, optimization=None, bits=30
    ):
        self._bits = 64  # max value of bits
        super().__init__(domain, number_points, seed, scramble, optimization)
        self.bits = bits

    @property
    def bits(self):
        """{1, 2}: Whether to create an orthogonal array of points"""
        return self._bits

    @bits.setter
    def bits(self, value):
        bits = int(value)
        if bits > 64:
            raise ValueError("bits max value is 64.")
        if 2**bits < self.number_points:
            raise ValueError("2**bits must be greater than number of points.")
        if self._bits != bits:
            self._bits = bits
            self._valid_cache = False

    @property
    def number_points(self):
        """int: random seed for generating points"""
        return self._number_points

    @number_points.setter
    def number_points(self, value):
        number_points = int(value)
        if numpy.ceil(numpy.log2(number_points)) != numpy.floor(
            numpy.log2(number_points)
        ):
            raise ValueError("Number of points must be power of 2")
        if number_points > 2**self.bits:
            raise ValueError("Number of points must be less than 2**bits")
        if self._number_points != number_points:
            self._number_points = number_points

            self._valid_cache = False

    def _get_random_points(self):
        """Generate a set of random points

        Generates a set of quasi-monte carlo Sobol Sequence points.
        """
        lower_bounds = [bound[0] for bound in self.domain]
        upper_bounds = [bound[1] for bound in self.domain]
        p_gen = scipy.stats.qmc.Sobol(
            self.num_dimensions,
            scramble=self.scramble,
            bits=self.bits,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.number_points)
        return scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class PointSetProduct(MultidimensionalPointSet):
    """Generate a grid using combinations of a unidimensional point set

    Parameters
    ----------
    point_sets : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def __init__(self, point_sets):
        super().__init__()
        self._point_sets = point_sets
        self._indexes = []

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return numpy.array([up.domain for up in self.point_sets])

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise TypeError("Domain must have size (num_dimensions, 2)")
        if any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("Lower bound must be less than upper bound")
        if len(self.point_sets) != domain.shape[0]:
            raise IndexError("Domain does not match number of point sets")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            for ps,d in zip(self.point_sets,domain):
                ps.domain = d
            self._valid_cache = False

    @property
    def point_sets(self):
        """:class:UnidimensionalPointSet: set of unique points"""
        return self._point_sets

    @property
    def indexes(self):
        """list: Grid points' indexes."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._indexes


class TensorProductPointSet(PointSetProduct):
    """Generate a grid using all combinations of unidimensional point sets

    Depending on the dimensionality, points and provided by
    :class:`UnidimensionalPointSet`, make full tensor grids.
    :meth:`generates_points` generates a full tensor grid

    Parameters
    ----------
    unique_points : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _create(self):
        """Generating the tensor grid

        Using the parameters of the class, generate a grid set using all
        combinations of unidimensional points
        """
        self._points = numpy.array(list(itertools.product(*self._point_sets)))


class SmolyakSparseProductPointSet(PointSetProduct):
    """Generate a grid using sparse combinations of unidimensional point sets

    Depending on the dimensionality, points and provided by
    :class:`UnidimensionalPointSet`, make full tensor grids.
    :meth:`generates_points` generates a full tensor grid

    Parameters
    ----------
    unique_points : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _create(self):
        """Generating the tensor grid

        Using the parameters of the class, generate a grid set using all
        combinations of unidimensional points
        """
        grid_points = None
        max_num_level = max([ob.num_levels for ob in self._point_sets])
        max_num_levels = [ob.num_levels for ob in self._point_sets]
        # get the combinations of levels
        index_composition = []
        for sum_of_levels in range(
            self.num_dimensions, self.num_dimensions + max_num_level
        ):
            index_composition.extend(
                list(
                    generate_compositions(
                        sum_of_levels, self.num_dimensions, include_zero=False
                    )
                )
            )
        index_composition = numpy.array(index_composition) - 1
        if min(max_num_levels) != max_num_level:
            # only check if point sets have different number of levels
            valid_comb = numpy.all(
                numpy.greater_equal(max_num_levels, index_composition), axis=1
            )
            index_composition = index_composition[valid_comb, :]
        for index_comp in index_composition:
            level_composition_index = [
                self._point_sets[d].level(index) for d, index in enumerate(index_comp)
            ]
            grid_points_ = numpy.array(
                numpy.meshgrid(*level_composition_index)
            ).T.reshape(-1, self.num_dimensions)
            if grid_points is None:
                grid_points = grid_points_
            else:
                grid_points = numpy.concatenate((grid_points, grid_points_), axis=0)

        # turn level combinations into points
        self._points = grid_points


def generate_compositions(value, num_parts, include_zero):
    """Generate compositions of a value into num_parts parts.

    The algorithm that is being used is NEXCOM and can be found in
    "Combinatorial Algorithms For Computers and Calculators",
    Second Edition, 1978.
    Authors: ALBERT NIJENHUIS and HERBERT S. WILF.
    https://doi.org/10.1016/C2013-0-11243-3
    ``include_zero`` parameter determines whether the compositions
    that contain zeros are parts of the output or not.
    The first composition will be (``value``, 0, ..., 0)
    (or (``value-num_parts``, 0, ..., 0) if ``include_zero`` is ``False``).
    Next, the first component (index = 0) is dropped by 1, and next component
    (index = 1) is incremented by 1. This goes on until index 0 reaches ``0``.
    Once, generated all compositions, the  next component is incremented
    by 1 and is fixed until all compositions are generated with the same
    method. This goes on until the last component reaches the ``value``
    (or ``value - num_parts`` if ``include_zero`` is ``False``).
    It is important to note that if ``include_zero`` is ``False``,
    all components will be incremented by 1.

    Parameters
    ----------
    value: int
        Value.
    num_parts: int
        Number of parts.
    include_zero : bool
        True if compositions contain zero, False otherwise.

    Yields
    ------
    list
        All possible compositions of the value into num_parts.

    Raises
    ------
    ValueError
        Number of parts cannot be greater than value if the desired output
        does not include compositions containing zeroes.
    """
    if value < num_parts and include_zero is False:
        raise ValueError(
            "When include_zero is {}, num_parts cannot be greater"
            " than the value".format(False)
        )
    value = value if include_zero else value - num_parts

    # (A) first entry
    r = [0] * num_parts
    r[0] = value
    t = value
    h = 0
    yield list(r) if include_zero else (numpy.array(r) + 1).tolist()

    # (D)
    while r[num_parts - 1] != value:
        # (B)
        if t != 1:
            h = 0

        # (C)
        h += 1
        t = r[h - 1]
        r[h - 1] = 0
        r[0] = t - 1
        r[h] += 1
        yield list(r) if include_zero else (numpy.array(r) + 1).tolist()
