import numpy

from .benchmark import BenchmarkFunction
from .rosenbr import _rosenbrock

class _wood(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] - v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 100. * v[...,0]
        v[...,0] = 1. - x[...,0]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = x[...,2] * x[...,2]
        v[...,0] = x[...,3] - v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = 90. * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 1. - x[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = -1. + x[...,1]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 10.1 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = -1. + x[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 10.1 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = -1. + x[...,1]
        v[...,0] = 19.8 * v[...,2]
        v[...,2] = -1. + x[...,3]
        v[...,3] = v[...,0] * v[...,2]
        v[...,1] += v[...,3]
        return v[...,1]


class hs001(_rosenbrock):
    @property
    def domain(self):
        return [[-9.0000000086, 10.9999999914], [-1.5, 10.9999999828]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0.9999999914, 0.9999999828]

class hs002(_rosenbrock):
    @property
    def domain(self):
        return [[-8.7756292513, 11.2243707487], [1.5, 11.5]]

    @property
    def global_minimum(self):
        return 0.0504261879
    
    @property
    def global_minimum_location(self):
        return [1.2243707487, 1.5]

class hs003(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [2])
        v[...,0] = x[...,1] - x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = 1.e-05 * v[...,1]
        rv = v[...,0] + x[...,1]
        return rv

class hs004(BenchmarkFunction):
    @property
    def domain(self):
        return [[1, 11], [0, 10]]

    @property
    def global_minimum(self):
        return 2.6666666667
    
    @property
    def global_minimum_location(self):
        return [1, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [2])
        v[...,0] = 1. + x[...,0]
        v[...,1] = pow(v[...,0], 3.)
        v[...,0] = 0.3333333333333333 * v[...,1]
        rv = v[...,0] + x[...,1]
        return rv


class hs005(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.5, 4.0], [-3.0, 3.0]]

    @property
    def global_minimum(self):
        return -1.913222955
    
    @property
    def global_minimum_location(self):
        return [-0.5471975512, -1.5471975512]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] + x[...,1]
        v[...,1] = numpy.sin(v[...,0])
        v[...,0] = x[...,0] - x[...,1]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,1] += 1.
        rv = v[...,1] + -1.5*x[...,0]
        rv += 2.5*x[...,1]
        return rv

class hs3mod(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [2])
        v[...,0] = -x[...,0]
        v[...,1] = v[...,0] + x[...,1]
        v[...,0] = v[...,1] * v[...,1]
        rv = v[...,0] + x[...,1]
        return rv

class hs038(_wood):
    @property
    def domain(self):
        return [[-10.0, 10.0], [-9.0, 11.0], [-9.0000000001, 10.9999999999],
                [-9.0000000001, 10.9999999999]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [1, 1, 0.9999999999, 0.9999999999]

class hs045(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]

    @property
    def global_minimum(self):
        return 1
    
    @property
    def global_minimum_location(self):
        return [1, 2, 3, 4, 5]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [2])
        v[...,0] = x[...,0] * x[...,1]
        v[...,1] = v[...,0] * x[...,2]
        v[...,0] = v[...,1] * x[...,3]
        v[...,1] = v[...,0] * x[...,4]
        v[...,0] = -0.008333333333333333 * v[...,1]
        v[...,1] = v[...,0] + 2.
        return v[...,1]
