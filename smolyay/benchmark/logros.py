import numpy

from .benchmark import BenchmarkFunction

class logros(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 11], [0, 11]]

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] - v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 10000. * v[...,0]
        v[...,1] += 1.
        v[...,0] = 1. - x[...,0]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = numpy.log(v[...,1])
        return v[...,2]
