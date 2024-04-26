import numpy

from .benchmark import BenchmarkFunction


class fermat_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.9999999999, 12.0000000001],
                [-8.8452994616, 11.1547005384], [1e-08, 10.00000001]]

    @property
    def global_minimum(self):
        return 7.4641016251
    
    @property
    def global_minimum_location(self):
        return [2.0000000001, 1.1547005384, 1e-08]

    def _function(self, x):
        return (
            numpy.sqrt(numpy.square(x[..., 2]) + numpy.square(x[..., 0]) + numpy.square(x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 2]) + numpy.square((-4) + x[..., 0]) + numpy.square(x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 2]) + numpy.square((-2) + x[..., 0]) + numpy.square((-4) + x[..., 1]))
            + x[..., 2]
        )


class fermat2_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8, 12], [-9.00000002, 10.99999998], [1e-08, 10.00000001]]

    @property
    def global_minimum(self):
        return 4.4721359695
    
    @property
    def global_minimum_location(self):
        return  [2, 0.99999998, 1e-08]

    def _function(self, x):
        return (
            numpy.sqrt(numpy.square(x[..., 2]) + numpy.square(x[..., 0]) + numpy.square(x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 2]) + numpy.square((-4) + x[..., 0]) + numpy.square(x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 2]) + numpy.square((-2) + x[..., 0]) + numpy.square((-1) + x[..., 1]))
            + x[..., 2]
        )
