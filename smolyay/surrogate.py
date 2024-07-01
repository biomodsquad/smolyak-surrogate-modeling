import abc

import numpy
import sklearn

class Surrogate(abc.ABC):
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    sampling method, and basis functions, a surrogate model can be generated
    that approximates a set of data.
    ``domain`` is the domain of the function to be approximated.
    :meth:`train`, generates a trained surrogate model.
    ``points`` stores the points that are used for sampling.
    :meth:`train_from_data` computes the coefficients based on
    a set of data (function values at transformed grid points).
    Once the surrogate is constructed, one can evaluate the surrogate
    through :meth:`__call__(x)`.
    :meth:`_reset_surrogate` resets the surrogate's coefficients,
    sampled points, original ``grids``(holds the grid points
    and their corresponding basis functions depending on the
    dimensionality and the generator), and ``data`` (function at
    sampling points).

    Parameters
    ----------
    point_set: MultidimensionalPointSet
        a point set of samples
    """

    def __init__(self):
        self._data = None
        self._point_set = None
        self._valid_cache = False

        
    @property
    def data(self):
        """list: data at sampling grid points."""
        if self._data is not None:
            return self._data.tolist()
        else:
            return None

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return self.domain.shape[0]
    
    @property
    def point_set(self):
        """MultidimensionalPointSet: points to be sampled"""
        return self._point_set
    
    @point_set.setter
    def point_set(self, value):
        self._point_set = value
        self._valid_cache = False

    @property
    def domain(self):
        """list: domain of the function to be approximated."""
        return self.point_set.domain

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if not numpy.array_equal(self._domain, domain):
            self._point_set.domain = domain
            self._valid_cache = False

    @property
    def points(self):
        """numpy.ndarray: Points to be sampled."""
        return self.point_set.points

    @abc.abstractmethod
    def predict(self, x):
        """Evaluate surrogate at a given input.

        Parameter
        ---------
        x: list (for dimension > 1), or float (for dimension = 1)
            Input.

        Returns
        -------
        float
            Surrogate output at x.

        Raises
        ------
        ValueError
            For surrogate to be evaluated, function needs to be trained.
        IndexError
            Input of the surrogate must be of length of the
            function's dimensionality.
        ValueError
            Input must lie in domain of surrogate.
        """
        pass

    def train(self, function):
        """Fit surrogate's components (basis functions) to the function.

        Parameters
        ----------
        function: callable
            Function to be approximated.
        """
        data = [function(point) for point in self.points]
        self.fit(data, self.points, data)

    @abc.abstractmethod
    def fit(self, x, y = None):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        x : list
            points that are sampled

        y : list
            function at grid points.

        Raises
        ------
        IndexError
            x should be of length of the y.
        """
        pass