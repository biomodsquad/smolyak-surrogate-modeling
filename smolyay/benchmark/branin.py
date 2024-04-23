import numpy

from .benchmark import BenchmarkFunction

class branin(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5, 10], [0, 15]]

    @property
    def global_minimum(self):
        return 0.3978873577
    
    @property
    def global_minimum_location(self):
        return [ 9.4247779642, 2.4750000028 ]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 0.12918450914398066 * v[...,0]
        v[...,0] = x[...,1] - v[...,1]
        v[...,1] = 1.5915494309189535 * x[...,0]
        v[...,2] = v[...,0] + v[...,1]
        v[...,0] = -6. + v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = numpy.cos(x[...,0])
        v[...,1] = 9.602112642270262 * v[...,0]
        v[...,2] += v[...,1]
        v[...,2] += 10.
        return v[...,2]
