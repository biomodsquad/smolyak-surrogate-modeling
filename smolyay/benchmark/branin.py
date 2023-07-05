import numpy

from .benchmark import BenchmarkFunction

class branin(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5, 10], [0, 15]]
        
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
