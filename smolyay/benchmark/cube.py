import numpy

from .benchmark import BenchmarkFunction

class _cube(BenchmarkFunction): 
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = -1. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = pow(x[...,0], 3.)
        v[...,2] = x[...,1] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 100. * v[...,0]
        v[...,0] = v[...,1] + v[...,2]
        return v[...,0]

class cube(_cube):
    @property
    def domain(self):
        return [[-18, 9.9], [-18, 9.9]]
