import abc

class BasisFunction(abc.ABC):
    """Abstract class for basis functions.

    Creates a framework for basis functions that are the building blocks
    for the terms in the surrogate model. 
    ``max_exactness`` represents the maximum level of exactness, or 
    accuracy, of a surrogate function this class can provide enough 
    information for. This class can provide enough information for 
    constructing a surrogate function of any accuracy at or below 
    max_exactness.
    ``num_extrema_per_level`` property represents the number of unique 
    extrema that correspond to increasing levels of exactness.
    ``extrema`` property represents all the unique extrema of the 
    basis function in order of computation. Each extrema is associated 
    with a Smolyak index described by the class IndexGrid, and the 
    Smolyak index of a given extrema is equal to the extrema's position
    in the array extrema.
    ``extrema_per_level`` property gives the Smolyak indexes for each
    grid level based on the num_extrema_per_level

    Parameters
    ----------
    max_exactness : int
        level of exactness class will describe

    """

    def __init__(self,max_exactness):
        self._max_exactness = max_exactness
        self._extrema = []
        self._num_extrema_per_level = []
        self._extrema_per_level = []
        self._update()

    @property
    def max_exactness(self):
        """Maximum level of exactness described"""
        return self._max_exactness

    @property
    def extrema(self):
        """Extrema assigned to Smolyak indices"""
        return self._extrema

    @property
    def num_extrema_per_level(self):
        """Number of extrema per grid level"""
        return self._num_extrema_per_level

    @property
    def extrema_per_level(self):
        """extrema in each grid level by index in property extrema"""
        return self._extrema_per_level

    @max_exactness.setter
    def max_exactness(self, max_exactness):
        """Set maximum level of exactness
        Sets maximmum level of exactness class instance will describe, will
        extend or truncate extrema_list and extrema_per_level_num to match
        new value

        Parameters
        ----------
        max_exactness : int
            new level of exactness class should describe
        """
        if max_exactness > self._max_exactness:
            # extend
            self._max_exactness = max_exactness
            # update other properties
            self._update()
        elif max_exactness < self._max_exactness:
            # truncate
            self._max_exactness = max_exactness
            # update other properties
            extrema_keep = sum(
                    self._num_extrema_per_level[:self._max_exactness+1])
            self._extrema = self._extrema[:extrema_keep]
            self._num_extrema_per_level = self._num_extrema_per_level[
                    :self._max_exactness+1]
            self._extrema_per_level = self._extrema_per_level[
                    :self._max_exactness+1]


    @abc.abstractmethod
    def _update(self):
        """Update properties described by level of exactness
        Compute extrema of basis function and the number of extrema
        associated with each level of exactness required to describe 
        the level of exactness in property self.max_exactness
        """
        pass

    @abc.abstractmethod
    def basis(self,x,n):
        """Compute term of basis function
        Returns the output of an nth degree basis function with input x

        Parameters
        ----------
        x : float
            input

        n : int
            degree

        Returns
        -------
        y : float
            output
        """
        pass




