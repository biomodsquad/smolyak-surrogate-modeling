import numpy

from .benchmark import BenchmarkFunction

class maratosb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.000000125, 8.0999998875], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,2] = v[...,0] + v[...,1]
        v[...,0] = -1. + v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = 1.e+06 * v[...,2]
        rv = v[...,0] + x[...,0]
        return rv
