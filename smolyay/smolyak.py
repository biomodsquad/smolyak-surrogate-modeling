import numpy as np
import scipy.special


class IndexGrid:
    """Generate index of grid points.

    Attributes
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

        if len(index_per_level) < exactness + 1:
            raise IndexError(
                "index_per_level must be an array with a"
                "length of at least {}"
                .format(exactness + 1))

    def level_indexes(self):
        """Generate indexes of levels.

        Returns
        -------
        level_indexes : list
            Unique indexes at each level.

        """
        # creating an array to store unique indexes in each level
        level_indexes = [[0]]*(self.exactness + 1)
        max_index = sum(self.index_per_level[0:self.exactness + 2])
        indexes = np.linspace(0, max_index, max_index+1, dtype=np.int32)

        # intial level
        level_indexes[0] = indexes[0:self.index_per_level[0]]

        # storing unique indexes for each level
        summ_index = self.index_per_level[0]
        for idx_index_per_level in range(1, self.exactness + 1):
            summ_index += self.index_per_level[idx_index_per_level]
            level_indexes[idx_index_per_level] = (
                indexes[summ_index-self.index_per_level[idx_index_per_level]:
                        summ_index])

        return level_indexes

    def grid_point_index(self):
        """Generate index of grid points.

        First, it generates allowed compositons of levels
        via NEXCOM algorithm. Then, it replaces the levels
        with their corresponding indexes, and
        makes the index for the grid points.

        Returns
        -------
        grid_points_index : numpy array
            Index of grid points

        """
        num_grid_points = 0
        level_indexes = self.level_indexes()
        level_compositions = []

        # compute the allowed level compositions
        for summ in range(self.dimension, self.exactness+self.dimension+1):

            # NEXCOM
            max_level_index = summ - self.dimension
            num_output = scipy.special.comb(max_level_index+self.dimension-1,
                                            self.dimension-1, exact=True)
            compositions = np.zeros((num_output, self.dimension),
                                    dtype=np.int32)

            # (A) first entry
            r = np.zeros(self.dimension, dtype=np.int32)
            r[0] = max_level_index
            t = max_level_index
            h = 0
            compositions[0] = r
            index = 1

            # (D): these termination conditions should be redundant
            while (r[self.dimension-1] != max_level_index
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

            # precompute the number of grid points
            # and replace the levels with indexes
            for composition in compositions:
                num_grid_point = 1
                level_compositions.append([])
                for index_i in range(self.dimension):
                    num_grid_point *= (
                        self.index_per_level[composition[index_i]])
                    level_compositions[-1].append(
                        level_indexes[composition[index_i]])
                num_grid_points += num_grid_point

        # create a numpy array with the size of grid points
        grid_points_index = np.zeros((
            num_grid_points, self.dimension), dtype=np.int32)
        num_grid_point = 0
        for level_index_composition in level_compositions:
            len_level = []
            # generate a list containing the number of indexes
            # corresponding to each level
            for level_index in level_index_composition:
                len_level.append(len(level_index))
            # generate grid point indexes
            for index_grid_i in np.ndindex(tuple(len_level)):
                for dimension_i in range(self.dimension):
                    index = index_grid_i[dimension_i]
                    grid_points_index[num_grid_point][dimension_i] = (
                        level_index_composition[dimension_i][index])
                num_grid_point += 1

        return grid_points_index
