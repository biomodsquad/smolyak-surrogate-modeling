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
        self._points = None
        self._valid_cache = False

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return numpy.shape(self.domain)[0]

    @property
    @abc.abstractmethod
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        pass

    @property
    def points(self):
        """numpy.ndarray: Points to be sampled."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._points

    def __len__(self):
        return self.points.shape[0]

    @staticmethod
    def _scale_to_domain(x, old, new):
        """Transform the points into new domain.

        Parameters
        ----------
        x: list
            point(s) to be transformed.
        old: list
            old domain.
        new: list
            new domain.

        Returns
        -------
        numpy.ndarray
            Transformed point(s).

        Raises
        ------
        TypeError
            Old and new domain must have the same shape.
            Domain should be a dim x 2 array.
        """
        old = numpy.array(old, copy=False, ndmin=2)
        new = numpy.array(new, copy=False, ndmin=2)

        # error checking
        if old.shape != new.shape:
            raise TypeError("Old and new domain must have the same shape")
        if old.ndim != 2 or old.shape[1] != 2:
            raise TypeError("Domain should be a dim x 2 array")

        new_x = new[:, 0] + (new[:, 1] - new[:, 0]) * (
            (x - old[:, 0]) / (old[:, 1] - old[:, 0])
        )
        # clamp bounds
        if len(new_x.shape) == 1:
            numpy.clip(new_x, new[:, 0], new[:, 1], out=new_x)
        else:
            for i in range(new.shape[0]):
                numpy.clip(new_x[..., i], new[i, 0], new[i, 1], out=new_x[..., i])
        return new_x

    @abc.abstractmethod
    def _create(self):
        """Abstract method for generating the mulitdimensional grid set"""
        pass


class RandomPointSet(MultidimensionalPointSet):
    """Point Set that uses randomly generated points

    A Multidimensional point set that creates a grid using a Monte Carlo
    or Quasi Monte Carlo method for generating points. This generation
    relies on the QMCEngine objects in the scipy.stats.qmc module.

    In addition to the domain of the grid, this class has 3 required and
    1 optional parameter to initailize and use the QMCEngines.
    `number_points` is the number of points to be generated. Certain methods
    of generating random number
    `method` is the Monte Carlo method of generated points. There are 4
    valid options for the method:
        "latin" - short for Latin Hypercube, generates points with
        scipy.stats.qmc.LatinHypercube object.
        "halton" - short for Halton Sequences, generated points with
        scipy.stats.qmc.Halton object.
        "sobol" - short for Sobol Sequences, generated points with
        scipy.stats.qmc.Sobol object. number_points must be a power of 2.
        "uniform" - use numpy.random.default_rng().uniform function
    The valid possible values of `method` are controlled by the property
    `valid_methods`.
    `seed` is the int or numpy.random.Generator instance is used to control the
    creation of pseudorandom numbers. This parameter is required to ensure
    an instance of random grid points can be recreated.
    `options` is an optional parameter for any parameters to pass to the
    QMCEngine that are not listed, and only affect the "latin", "halton", and
    "sobol" methods. These possible parameters are:

    scramble : bool
        Default True. Applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points.
    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.
    strength : {1, 2}
        Default 1. Exclusive to Latin Hypercube. If 2, produces latin hypercube
        points based on an orthogonal array. number_points is constrained to
        squares of prime number whose square root is one greater than num_dimensions.
    bits : int
        Default 30. Exclusive to Sobol. Is the number of bits used by the
        generator.


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
        domain = numpy.sort(numpy.array(value, ndmin=2),axis=1)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise TypeError("Domain must have size (num_dimensions, 2)")
        if any(domain[:,0] >= domain[:,1]):
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
        if not isinstance(value,numpy.random.Generator):
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
        Carlo method. Available methods are latinhypercube, Halton sequences,
        and Sobol sequences. If no method is selected, random points will be
        generated using numpy.random's rand function.

        Parameters
        ----------
        domain: list
            domain of the random points

        number_points: int
            number of points to generate

        seed: {int, numpy.random.Generator}
            seed for generating the random points

        Returns
        -------
        numpy.ndarray
            generated random points

        Raises
        ------
        ValueError
            number of points for Sobol sequences must be a power of 2
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
    """Generates a point set using a QMCEngine"""

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
    """Generates a grid using a uniform distribution"""

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
    """Generates a grid using a uniform distribution"""

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
    """Generates a grid using a uniform distribution"""

    def __init__(
        self, domain, number_points, seed, scramble=True, optimization=None, bits=30
    ):
        super().__init__(domain, number_points, seed, scramble, optimization)
        self._bits = 30

        self.bits = bits

    @property
    def bits(self):
        """{1, 2}: Whether to create an orthogonal array of points"""
        return self._bits

    @bits.setter
    def bits(self, value):
        bits = int(value)
        if bits > 64:
            raise TypeError("bits cannot exceed 64.")
        if self._bits != bits:
            self._bits = bits
            self._valid_cache = False

    def _get_random_points(self):
        lower_bounds = [bound[0] for bound in self.domain]
        upper_bounds = [bound[1] for bound in self.domain]
        if numpy.ceil(numpy.log2(self.number_points)) != numpy.floor(
            numpy.log2(self.number_points)
        ):
            raise ValueError("Number of points must be power of 2")
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
