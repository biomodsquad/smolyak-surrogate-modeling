#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:00:15 2022

@author: mzf0069
"""

import numpy
import scipy.special


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
        self._needs_update = True

    def _update(self):
        """Update the indexes of grid points.

        It generates the indexes for each level
        depending on the index_per_level, and makes the
        indexes of grid points.

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

        points = None
        # get all combinations of points at each level
        for summ in range(self.dimension, self.exactness+self.dimension+1):
            compositions = generate_compositions(summ, self.dimension)
            for composition in compositions:
                point_ids = [level_indexes[i] for i in composition]
                points_ = numpy.array(
                    numpy.meshgrid(*point_ids)).T.reshape(-1,
                                                          self.dimension)
                if points is None:
                    points = points_
                else:
                    points = numpy.concatenate((points, points_), axis=0)
        self._grid_point_index = points
        self._needs_update = False

    @property
    def level_indexes(self):
        """Generate indexes of levels.

        Returns
        -------
        level_indexes : list
            Unique indexes at each level.

        """
        if self._needs_update:
            self._update()

        return self._level_indexes

    @property
    def grid_point_index(self):
        """Generate index of grid points.

        It generates allowed compositons of levels,
        and depending on the indexes for each level,
        it makes the indexes of the grid points.

        Returns
        -------
        grid_points_index : numpy array
            Index of grid points

        """
        if self._needs_update:
            self._update()

        return self._grid_point_index


def generate_compositions(summ, dimension):
    """Genearte allowed compositions of levels using NEXCOM algorithm.

    Parameters
    ----------
    summ: int
        Summation of levels.
    dimension: int
        Number of independent variables.

    Returns
    -------
    compositions: list
        Allowed composition of levels.
    """
    max_level_index = summ - dimension
    num_output = scipy.special.comb(max_level_index+dimension-1,
                                    dimension-1, exact=True)
    compositions = numpy.zeros((num_output, dimension),
                               dtype=numpy.int32)

    # (A) first entry
    r = numpy.zeros(dimension, dtype=numpy.int32)
    r[0] = max_level_index
    t = max_level_index
    h = 0
    compositions[0] = r
    index = 1

    # (D): these termination conditions should be redundant
    while (r[dimension-1] != max_level_index
           and index < num_output + 1):

        # (B)
        if t != 1:
            h = 0

        # (C)
        h += 1
        t = r[h-1]
        r[h-1] = 0
        r[0] = t-1
        r[h] += 1

        # (D)
        compositions[index] = r
        index += 1

    return compositions

