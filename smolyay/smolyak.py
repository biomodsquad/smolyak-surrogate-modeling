import numpy


class IndexGrid:
    """Indexes of grid points for constructing a surrogate function.

    Parameters
    ----------
    dimension : int
        Number of independent variables.
    exactness : int
        Level of exactness of the approximation.
    index_per_level : list
        Number of indexes at each level.
    """

    def __init__(self, dimension, exactness, index_per_level):
        self.dimension = dimension
        self.exactness = exactness
        self.index_per_level = index_per_level

    @property
    def dimension(self):
        """int: Dimensionality."""
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value
        self._needs_update = True

    @property
    def exactness(self):
        """int: Level of exactness."""
        return self._exactness

    @exactness.setter
    def exactness(self, value):
        self._exactness = value
        self._needs_update = True

    @property
    def index_per_level(self):
        """list: Index per level."""
        return self._index_per_level

    @index_per_level.setter
    def index_per_level(self, value):
        self._index_per_level = value
        self._needs_update = True

    @property
    def level_indexes(self):
        """Generate indexes of levels."""
        if self._needs_update:
            self._update()

        return self._level_indexes

    @property
    def grid_point_indexes(self):
        """Generate index of grid points."""
        if self._needs_update:
            self._update()

        return self._grid_point_indexes

    def _update(self):
        """Update the indexes of grid points.

        Generates the indexes for each level
        depending on the index_per_level, and makes the
        indexes of grid points.

        Raises
        ------
        IndexError
            Index per level must be an array with a
            length of at least exactness + 1.
        """
        # requirement for generating the grid points
        if len(self.index_per_level) < self.exactness + 1:
            raise IndexError(
                "index_per_level must be an array with a"
                " length of at least {}"
                .format(self.exactness + 1))

        # cumulative sum to get end index of each level, up to exactness+1
        end_levels = numpy.cumsum(self.index_per_level[:self.exactness+1])
        # create ranges of indexes at each level
        level_indexes = [list(range(end-n, end))
                         for end, n in zip(end_levels, self.index_per_level)]
        self._level_indexes = level_indexes

        # get all combinations of points at each level
        grid_points_idx = None
        for sum_of_levels in range(self.dimension,
                                   self.dimension+self.exactness+1):
            for composition in generate_compositions(
                    sum_of_levels, self.dimension, include_zero=False):
                # indexes start from zero
                index_composition = composition - 1
                # generate all combinations of the arrays along each dimension
                level_cmp_idx = [level_indexes[idx]
                                 for idx in index_composition]
                grid_points_idx_ = numpy.array(
                    numpy.meshgrid(*level_cmp_idx)).T.reshape(-1,
                                                              self.dimension)
                if grid_points_idx is None:
                    grid_points_idx = grid_points_idx_
                else:
                    grid_points_idx = numpy.concatenate(
                        (grid_points_idx, grid_points_idx_), axis=0)
        self._grid_point_indexes = grid_points_idx
        self._needs_update = False


def generate_compositions(value, num_parts, include_zero):
    """Genearte compositions of a value into num_parts parts.

    The algorithm that is being used is NEXCOM and can be found in
    "Combinatorial Algorithms For Computers and Calculators",
    Second Edition, 1978.
    Authors: ALBERT NIJENHUIS and HERBERT S. WILF.
    https://doi.org/10.1016/C2013-0-11243-3

    include_zero parameter determines whether the compositions
    that contain zeros are parts of the output or not.

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
    numpy array
        All possible compositions of the value into num_parts.
    """
    value = value if include_zero else value - num_parts

    # (A) first entry
    r = numpy.zeros(num_parts, dtype=numpy.int32)
    r[0] = value
    t = value
    h = 0
    yield r if include_zero else r+1

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
        yield r if include_zero else r+1
