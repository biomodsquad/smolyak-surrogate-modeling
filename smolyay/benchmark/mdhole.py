import numpy

from .benchmark import BenchmarkFunction

class mdhole(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [-10, 10]]
        
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = -x[...,1]
        v[...,1] = numpy.sin(x[...,0])
        v[...,2] = v[...,0] + v[...,1]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 100. * v[...,0]
        rv = v[...,2] + x[...,0]
        return rv
