import numpy

from .benchmark import BenchmarkFunction

class nasty(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = 5.e+09 * x[...,0]
        v[...,1] = 1.e+10 * x[...,0]
        v[...,2] = v[...,0] * v[...,1]
        v[...,0] = 0.5 * x[...,1]
        v[...,1] = v[...,0] * x[...,1]
        v[...,0] = v[...,2] + v[...,1]
        return v[...,0]
