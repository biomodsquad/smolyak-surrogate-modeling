import numpy

from .benchmark import BenchmarkFunction

class rosenbr(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 5.0], [-10.0, 10.0]]

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,7))
        v[...,0] = x[...,0] * x[...,0]
        v[...,5] = -10. * v[...,0]
        v[...,5] = v[...,5] + 10.*x[...,1]
        v[...,6] = 1. - x[...,0]
        v[...,2] = v[...,5] * v[...,5]
        v[...,3] = v[...,6] * v[...,6]
        v[...,4] = v[...,2] + v[...,3]
        return v[...,4]
