import abc

import numpy

class BenchmarkFunction(abc.ABC):
    """Benchmark Function

    Implementation of benchmark problems used for optimization, testing, and 
    analysis of surrogate function solvers. These benchmark problems come from
    https://sahinidis.coe.gatech.edu/dfo

    These functions operate on a defined `domain` they can be evaluated on,
    and the upper and lower bounds of this domain can be the domain in which
    solutions exist or can be arbitrary.

    """
    def __call__(self,x):
        """Evaluate the function.

        Parameters
        ----------
        x : list
            Function input.

        Raises
        ------
        ValueError
            If the input is outside the function domain.
        """
        x = numpy.array(x, copy=False,ndmin=2)
        if self.dimension > 1:
            oob = any(numpy.any(xi < bound[0]) or numpy.any(xi > bound[1]) 
                    for xi,bound in zip(x.transpose(),self.domain))
        else:
            oob = (numpy.any(x < self.domain[0][0]) or 
                    numpy.any(x > self.domain[0][1]))
        if oob:
            raise ValueError("Input out domain of function.")
        return numpy.squeeze(self._function(x))
    
    @property
    def name(self):
        """Name of the function"""
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
 
    @abc.abstractmethod
    def _function(self,x):
        pass



