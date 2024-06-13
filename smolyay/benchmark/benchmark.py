import abc

import numpy


class BenchmarkFunction(abc.ABC):
    """Benchmark Function

    These functions operate on a defined `domain` they can be evaluated on,
    and the upper and lower bounds of this domain can be the domain in which
    solutions exist or can be arbitrary.

    """

    def __call__(self, x):
        """Evaluate the function.

        Parameters
        ----------
        x : list
            Function input.

        Raises
        ------
        IndexError
            If the shape of the input does not match dimension of domain
        ValueError
            If the input is outside the function domain.
        """
        x = numpy.array(x, copy=False, ndmin=2)
        if self.dimension == 1:
            if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1:
                # the cast to 2d puts these in wrong order, so transpose
                x = x.T
            else:
                x = x[..., numpy.newaxis]
        if x.shape[-1] != self.dimension:
            raise IndexError("Input must match dimension of domain")
        if any(
            numpy.any(x[..., i] < self.domain[i][0])
            or numpy.any(x[..., i] > self.domain[i][1])
            for i in range(self.dimension)
        ):
            raise ValueError("Input outside domain of function.")
        return numpy.squeeze(self._function(x))

    @property
    def name(self):
        """str: Name of the function"""
        return type(self).__name__

    @property
    def dimension(self):
        """int: Number of variables."""
        return len(self.domain)

    @property
    def lower_bounds(self):
        """list: the lower bounds of the domain of each variable."""
        return [bound[0] for bound in self.domain]

    @property
    def upper_bounds(self):
        """list: the upper bounds of the domain of each variable."""
        return [bound[1] for bound in self.domain]

    @property
    @abc.abstractmethod
    def domain(self):
        """list: Domain of the function.

        The domain must be specified as lower and upper bounds for each variable as a list of lists.
        """
        pass

    @property
    @abc.abstractmethod
    def global_minimum(self):
        """float: global minimum of the function."""
        pass

    @property
    @abc.abstractmethod
    def global_minimum_location(self):
        """list: location global minimum of the function."""
        pass

    @abc.abstractmethod
    def _function(self, x):
        pass
