import numpy

from .benchmark import BenchmarkFunction

class mexhat(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.1417744688, 9.77240297808], [-9.2634512042, 9.66289391622]]

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,4))
        v[...,0] = -1. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = -2. * v[...,1]
        v[...,1] = x[...,0] * x[...,0]
        v[...,2] = x[...,1] - v[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,2] = v[...,1] / 10000.
        v[...,2] += -0.02
        v[...,1] = -1. + x[...,0]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = v[...,2] * v[...,2]
        v[...,2] = 10000. * v[...,3]
        v[...,3] = v[...,0] + v[...,2]
        return v[...,3]
