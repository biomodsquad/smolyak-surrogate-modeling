import abc
import itertools

import numpy

from smolyay.basis import (BasisFunction, ChebyshevFirstKind,
                           BasisFunctionSet, NestedBasisFunctionSet)


class IndexGridGenerator(abc.ABC):
    """Grid points for making a surrogate.

    Depending on the approach and dimensionality
    , a set of grid points, and their corresponding
    basis functions are generated. These grid points
    and functions can then be used for approximation
    of complex systems.

    Parameters
    ----------
    basis_set: :class:`BasisFunctionSet`
        A set of unidimensional points and functions.

    Raises
    ------
    TypeError
        ``basis_set`` must be a :class:`BasisFunctionSet`.
    """

    def __init__(self, basis_set):
        self.basis_set = basis_set
        if not isinstance(basis_set, BasisFunctionSet):
            raise TypeError('Basis set must be a BasisFunctionSet')

    @property
    def basis_set(self):
        """:class:`BasisFunctionSet`: basis set (points and functions)."""
        return self._basis_set

    @basis_set.setter
    def basis_set(self, basis_set):
        self._basis_set = basis_set

    def __call__(self, dimension):
        """Make grid points and their corresponding basis functions.

        Parameters
        ----------
        dimension: int
            Number of idependent variables.

        Returns
        -------
        :class:`IndexGrid`
            Grid points (and their indexes) and their corresponding functions.
        """
        pass


class SmolyakGridGenerator(IndexGridGenerator):
    r"""grid points and basis functions for constructing a surrogate.

    Create grid points and their corresponding basis functions to sample
    complex functions through Smolyak sparse sampling method.
    ``basis_set`` is a :class:`NestedBasisFunctionSet` which contains
    one-dimensional nested points (points at each level of approximation
    contains the points from all previous levels of approximations), their
    corresponding basis functions, and levels, which indicates
    levels each constituent basis function was taken from.
    At this point, all the data extracted from ``basis_set``
    are unidimensional.

    For a given number of independent variables (``dimension``), the
    Smolyak method of sparse sampling, creates grid points, and their
    corresponding basis functions through sparse tensor product,
    meaning that more important elements are sampled rather than a
    full tensor product sampling. For instance, if there are four points,
    the full tensor product generates :math:4^n (n being the dimension).
    However, the Smolyak method only generates parts of the
    full tensor product's elements, accounting only for the most crucial
    n-dimensional points. The size of the n-dimensional points
    (grid points) is dependent on a parameter, exactness
    (level of approximation). As exactness increases, the approximation
    accuracy should increase.

    Through the Smolyak technique, each dimension is assigned an indicy which
    is integers from 1 to exactness + 1.

    ..math::
        n: dimensionality
        \mu: exactness
        [k_1, k_2, ..., k_n] = indices
        where k_{1,2,...,n} = [1, 2, ..., \mu +1]
        n <= k_1 + k_2 + ... + k_n <= n + \mu

    For instance:
    ..math::
        if n = 2 and \mu = 1, then:
            if \sum_{1}^{n} k_i = 2:
                (k_1 = 1, k_2 = 1)
            if \sum_{i=1}^{n} k_i = 3:
                (k_1 = 1, k_2 = 2)
                (k_1 = 2, k_2 = 1)

    An object of this class, generates the indexes of
    grid points (integer grids) depending on dimensionality and
    exactness. First, as above, all possible combinations of
    indices are computed, which then can be
    replaced with indexes based on the points' distribution at each
    indicy, determined by levels (each indicy is the index of the levels,
    extracted from ``basis_set``). Then, grid points' indexes are
    generated by making all combinations of indexes for
    each combination of levels.

    Once grid points' indexes are generated, actual grid points
    and their corresponding basis functions can be made by simply replacing
    the indexes with their corresponding points and basis functions.
    multi-dimensional levels (levels' indexes each basis function was taken
    from) are also generated.
    All the mentioned outputs are stored in a :class:`NestedIndexGrid`
    object.

    Parameters
    ----------
    basis_set: :class:`NestedBasisFunctionSet`
        Unidimensional points, basis functions and levels.

    Raises
    ------
    TypeError
        ``basis_set`` must be a :class:`NestedBasisFunctionSet`.

    Example
    -------
    Consider dimension (n) = 2, and exactness (\mu) = 1 and
    Chebyshev's polynomial of first kind, and its extremums as
    the basis function and points, then:
    ..math::
        levels in one-dimension = [[0], [1,2]]
        indices = [[1,1], [1,2], [2,1]]
    replacing each indicy with corresponding indexes determined by levels:
    ..math::
        level's combination = [([0],[0]), ([0],[1,2]), ([1,2],[0])]
        then:
        grid points indexes = [(0,0), (0,1), (0,2), (1,0), (2,0)]
        points in one-dimension = [0, -1, 1]
        basis in one-dimension = [:class:`ChebyshevFirstKind`,
                :class:`ChebyshevFirstKind`, :class:`ChebyshevFirstKind`]

    Then:
    ..math:
        grid points = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        levels in two dimension: [(0, 0), (0, 1), (0, 1), (1, 0), (1, 0)]
        grid points' functions = [(:class:`ChebyshevFunction`,
        :class:`ChebyshevFunction`)
        , (:class:`ChebyshevFirstKind`, :class:`ChebyshevFirstKind`),
        (:class:`ChebyshevFirstKind`, :class:`ChebyshevFirstKind`),
        (:class:`ChebyshevFirstKind`, :class:`ChebyshevFirstKind`),
        (:class:`ChebyshevFirstKind`, :class:`ChebyshevFirstKind`)]

    """

    def __init__(self, basis_set):
        super().__init__(basis_set)
        if not isinstance(self.basis_set, NestedBasisFunctionSet):
            raise TypeError('Basis set must be nested for Smolyak grid')

    def __call__(self, dimension):
        """Make grid points and their corresponding basis functions.

        Depending on the dimensionality, a set of grid points are generated
        (integer grid points). Grid points of the basis function, and
        their corresponding basis function and levels are then
        generated by replacing the indexes with their
        corresponding unidimensional points, functions, and levels' indexes.

        Parameters
        ----------
        dimension: int
            Number of independent variables.

        Returns
        -------
        :class:`NestedIndexGrid`
            Grid points (and their indexes),levels,
            and their corresponding functions.
        """
        grid_points_indexes = None
        for sum_of_levels in range(dimension,
                                   dimension+len(self._basis_set.levels)):
            for composition in generate_compositions(
                    sum_of_levels, dimension, include_zero=False):
                index_composition = numpy.array(composition) - 1
                # generate all combinations of
                # the arrays along each dimension
                level_composition_index = [self._basis_set.levels[index]
                                           for index in index_composition]
                grid_points_indexes_ = (numpy.array(
                    numpy.meshgrid(*level_composition_index))
                    .T.reshape(-1, dimension))
                if grid_points_indexes is None:
                    grid_points_indexes = grid_points_indexes_
                else:
                    grid_points_indexes = numpy.concatenate(
                        (grid_points_indexes,
                         grid_points_indexes_), axis=0)
        # make integer grids, levels, grid points and basis functions
        grid_points_indexes = grid_points_indexes.tolist()
        level_lookup = {point: level for level, points in
                        enumerate(self._basis_set.levels) for point in points}
        levels = numpy.array([level_lookup[index] for index in
                              numpy.concatenate(grid_points_indexes)]
                             ).reshape((len(grid_points_indexes),
                                        dimension)).tolist()
        grid_points = numpy.array(self._basis_set.points
                                  )[numpy.array(grid_points_indexes)].tolist()
        grid_points_basis = numpy.array(self._basis_set.basis_functions)[
            numpy.array(grid_points_indexes)].tolist()

        return NestedIndexGrid(grid_points_indexes, grid_points,
                               grid_points_basis, levels)


class TensorGridGenerator(IndexGridGenerator):
    """Create grid points and functions for constructing a surrogate.

    Depending on the dimensionality, points and basis functions provided
    by :class:`BasisFunctionSet`, the :meth:`__call__` makes full tensor
    grids.
    :meth:`__call__`  generates the indexes of
    grid points (integer grids) depending on dimensionality (full tensor grid).
    Grid points and their functions are made by replacing the indexes with
    corresponding point.
    """

    def __init__(self, basis_set):
        super().__init__(basis_set)

    def __call__(self, dimension):
        """Make grid points and their corresponding basis functions.

        Depending on the dimensionality, a set of grid points are generated
        (integer grid points). Grid points of the basis function, and
        their corresponding basis functions are then generated by replacing
        the indexes (integer numbers) with their points and functions.

        Parameters
        ----------
        dimension: int
            Number of independent variables.

        Returns
        -------
        :class:`IndexGrid`
            Grid points (and their indexes),their corresponding functions.
        """
        points_indexes = numpy.arange(len(self._basis_set.points))
        # make integer grids, grid points and basis functions
        grid_points_indexes = list(itertools.product(
            *[points_indexes for point in range(dimension)]))
        grid_points_indexes = list(map(list, grid_points_indexes))
        grid_points = numpy.array(self._basis_set.points
                                  )[numpy.array(grid_points_indexes
                                                )].tolist()
        grid_points_basis = (numpy.array(self._basis_set.basis_functions)[
            numpy.array(grid_points_indexes)].tolist())

        return IndexGrid(grid_points_indexes, grid_points,
                         grid_points_basis)


class IndexGrid:
    """Set of grid points and their corresponding functions (a data structure).

    A data structure that contains grid points (and their indexes)
    and their corresponding basis functions. This data structure
    can then be used to generate a surrogate for complex systems.

    Property ``indexes``, ``points``, and
    ``basis`` represent the indexes of grid points,
    actual grid points, and their basis functions.
    """

    def __init__(self, indexes, points, basis):
        self._indexes = indexes
        self._points = points
        self._basis = basis

    @property
    def indexes(self):
        """list: Grid points' indexes."""
        return self._indexes

    @property
    def points(self):
        """list: Grid points."""
        return self._points

    @property
    def basis(self):
        """list: Basis functions of the grid points."""
        return self._basis


class NestedIndexGrid(IndexGrid):
    """Set of grid points and their corresponding functions (a data structure).

    Generate a data structure that contains grid points (and their indexes)
    and their corresponding basis functions. This data structure
    can then be used to generate a surrogate for complex systems.

    Property ``indexes``, ``points``, and ``basis`` represent the nested
    integer grid points, nested grid points, and their basis functions.
    ``levels`` represent approximation level each basis function was taken
    from.
    """

    def __init__(self, indexes, points, basis,
                 levels):
        super().__init__(indexes, points, basis)
        self._levels = levels

    @property
    def levels(self):
        """list: level each grid points' basis function was taken from."""
        return self._levels


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
            " than the value"
            .format(False))
    value = value if include_zero else value - num_parts

    # (A) first entry
    r = [0]*num_parts
    r[0] = value
    t = value
    h = 0
    yield list(r) if include_zero else (numpy.array(r)+1).tolist()

    # (D)
    while r[num_parts-1] != value:
        # (B)
        if t != 1:
            h = 0

        # (C)
        h += 1
        t = r[h-1]
        r[h-1] = 0
        r[0] = t-1
        r[h] += 1
        yield list(r) if include_zero else (numpy.array(r)+1).tolist()
