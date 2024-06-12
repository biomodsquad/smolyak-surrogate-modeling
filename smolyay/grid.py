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
        if self._points is None:
            self._generate_points()
        return self._points

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
    def _generate_points(self):
        """Abstract method for generating the mulitdimensional grid set"""


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

    def __init__(self, domain, number_points, method, seed, options=None):
        super().__init__()
        self._domain = None
        self._number_points = None
        self._method = None
        self._seed = None
        self._options = None

        self.domain = domain
        self.number_points = number_points
        self.method = method
        self.seed = seed
        self.options = options

    @property
    def valid_methods(self):
        """list: the allowed methods for generating random numbers"""
        return ["latin", "halton", "sobol", "uniform"]

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return self._domain

    @domain.setter
    def domain(self, value):
        value = numpy.array(value, ndmin=2)
        if value.ndim != 2 or value.shape[1] != 2:
            raise IndexError("Domain must have size (num_dimensions, 2)")
        self._domain = value

    @property
    def number_points(self):
        """int: random seed for generating points"""
        return self._number_points

    @number_points.setter
    def number_points(self, value):
        if not isinstance(value, (int, numpy.random.Generator)):
            raise TypeError("number_points must be an integer or Generator.")
        self._number_points = value

    @property
    def method(self):
        """str: method of sampling"""
        return self._method

    @method.setter
    def method(self, value):
        if value.casefold() in self.valid_methods:
            self._method = value.casefold()
        else:
            raise ValueError("Method can only be latin, halton, sobol, or uniform.")

    @property
    def seed(self):
        """int: random seed for generating points"""
        return self._seed

    @seed.setter
    def seed(self, value):
        if not isinstance(value, int):
            raise TypeError("seed must be an integer.")
        self._seed = value

    @property
    def options(self):
        """None or dict: additional parameters passed to QMCEngine"""
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    def _generate_points(self):
        """Generating the Monte Carlo mulitdimensional grid set

        Using the parameters of the class, generate a grid set using
        the a Monte Carlo/Quasi Monte Carlo method.
        """
        self._points = self.get_random_points(
            self.domain, self.number_points, self.method, self.seed, self.options
        )

    @staticmethod
    def get_random_points(domain, number_points, method, seed, options=None):
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

        method: {"uniform", "halton", "sobol", "latin"}
            the method of generating random points

        seed: {int, numpy.random.Generator}
            seed for generating the random points

        options: {None, dict}, optional
            if method != "uniform", additional parameters passed to the QMCEngine

        Returns
        -------
        numpy.ndarray
            generated random points

        Raises
        ------
        ValueError
            number of points for Sobol sequences must be a power of 2
        """

        lower_bounds = [bound[0] for bound in domain]
        upper_bounds = [bound[1] for bound in domain]
        num_dimensions = len(lower_bounds)
        if isinstance(options, dict):
            args = {"seed": seed, **options}
        else:
            args = {"seed": seed}
        if method.casefold() == "uniform":
            p_gen = numpy.random.default_rng(seed=seed).uniform(
                size=(number_points, num_dimensions )
            )
        elif method.casefold() == "latin":
            p_gen = scipy.stats.qmc.LatinHypercube(num_dimensions , **args).random(
                n=number_points
            )
        elif method.casefold() == "halton":
            p_gen = scipy.stats.qmc.Halton(num_dimensions, **args).random(n=number_points)
        elif method.casefold() == "sobol":
            if numpy.ceil(numpy.log2(number_points)) != numpy.floor(
                numpy.log2(number_points)
            ):
                raise ValueError("Number of points must be power of 2")
            p_gen = scipy.stats.qmc.Sobol(num_dimensions, **args).random(n=number_points)
        else:
            raise ValueError("Invalid method")

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

    @point_sets.setter
    def point_sets(self, value):
        self._point_sets = value

    @property
    def indexes(self):
        """list: Grid points' indexes."""
        if self._indexes is None:
            self._generate_points()
        return self._indexes


class TensorPointSet(PointSetProduct):
    """Generate a grid using all combinations of unidimensional point sets

    Depending on the dimensionality, points and provided by
    :class:`UnidimensionalPointSet`, make full tensor grids.
    :meth:`generates_points` generates a full tensor grid

    Parameters
    ----------
    unique_points : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _generate_points(self):
        """Generating the tensor grid

        Using the parameters of the class, generate a grid set using all
        combinations of unidimensional points
        """
        self._points = numpy.array(list(itertools.product(*self._point_sets)))
        


class SmolyakPointSet(PointSetProduct):
    """Generate a grid using sparse combinations of unidimensional point sets

    Depending on the dimensionality, points and provided by
    :class:`UnidimensionalPointSet`, make full tensor grids.
    :meth:`generates_points` generates a full tensor grid

    Parameters
    ----------
    unique_points : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _generate_points(self):
        """Generating the tensor grid

        Using the parameters of the class, generate a grid set using all
        combinations of unidimensional points
        """
        grid_points = None
        max_num_level = max([ob.num_levels for ob in self._point_sets]) 
        max_num_levels = [ob.num_levels for ob in self._point_sets]
        # get the combinations of levels
        index_composition = []
        for sum_of_levels in range(self.num_dimensions, self.num_dimensions + max_num_level):
            index_composition.extend(
                list(generate_compositions(sum_of_levels, self.num_dimensions, include_zero=False))
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
                grid_points = numpy.concatenate(
                    (grid_points,
                        grid_points_), axis=0)

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
