from abc import ABC, abstractmethod

class BasisFunction(ABC):
    """Abstract class for basis functions.

    Creates a framework for basis functions that are the building blocks
    for the terms in the surrogate model. 
    ''max_exactness'' represents the maximum level of exactness, or 
    accuracy, of a surrogate function this class can provide enough 
    information for. This class can provide enough information for 
    constructing a surrogate function of any accuracy at or below 
    max_exactness.
    ''extrema_per_level_num'' property represents the number of unique 
    extrema that correspond to increasing levels of exactness. It is 
    assumed that for an exactness of 0 the number of unique extrema is 1.
    ''extrema_list'' property represents all the unique extrema of the 
    basis function in order of computation. Each extrema is associated 
    with a Smolyak index described by the class IndexGrid, and the 
    Smolyak index of a given extrema is equal to the extrema's position
    in the array extrema_list. It is assumed the first extrema of any
    basis function is 0.

    Parameters
    ----------
    max_exactness : int
        level of exactness class will describe

    """

    def __init__(self,max_exactness):
        """Initializer"""
        self._extrema_list = [0]
        self._extrema_per_level_num = [1]
        self._max_exactness = max_exactness
        self.update_extrema_index_per_level()

    @property
    def max_exactness(self):
        """Maximum level of exactness described"""
        return self._max_exactness

    @property
    def extrema_list(self):
        """Extrema assigned to Smolyak indices"""
        return self._extrema_list

    @property
    def extrema_per_level_num(self):
        """Number of extrema per grid level"""
        return self._extrema_per_level_num

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
        if self._max_exactness < max_exactness:
            # extend
            min_exactness = self._max_exactness
            self._max_exactness = max_exactness
            # update other properties
            self.update_extrema_index_per_level(min_exactness+1)
        elif self._max_exactness > max_exactness:
            # truncate
            self._max_exactness = max_exactness
            # update other properties
            extrema_keep = sum(
                    self._extrema_per_level_num[0:self._max_exactness+1])
            self._extrema_list = self._extrema_list[0:extrema_keep]
            self._extrema_per_level_num = self._extrema_per_level_num[
                    0:self._max_exactness+1]

    @abstractmethod
    def update_extrema_index_per_level(self, next_exactness = 1):
        """Updates properties described by level of exactness
        Compute extrema of basis function and the number of extrema
        associated with each level of exactness required to describe 
        the level of exactness in property self.max_exactness

        Parameters
        ----------
        next_exactness : int
            level of exactness one higher than what the class describes
            before updating
        """
        pass

    @abstractmethod
    def basis_fun(x,n):
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




