import numpy

from .benchmark import BenchmarkFunction


class median_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[1e-08, 10.00000001], [-9.499789331, 10.500210669]]

    @property
    def global_minimum(self):
        return 4.942409183
    
    @property
    def global_minimum_location(self):
        return [1e-08, 0.500210669]

    def _function(self, x):
        return (
            numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.171747132) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.843266708) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.550375356) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.301137904) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.292212117) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.224052867) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.349830504) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.856270347) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.067113723) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.500210669) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.998117627) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.578733378) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.991133039) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.762250467) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.130692483) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.639718759) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.159517864) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.250080533) + x[..., 1]))
            + numpy.sqrt(numpy.square(x[..., 0]) + numpy.square((-0.668928609) + x[..., 1]))
            + x[..., 0]
        )
