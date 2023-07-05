import numpy

from .benchmark import BenchmarkFunction

class sineval(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0000000002, 8.99999999982], [-10.0000000002, 8.99999999982]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = numpy.sin(x[...,0])
        v[...,1] = x[...,1] - v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 1000. * v[...,0]
        v[...,0] = x[...,0] * x[...,0]
        v[...,2] = 0.25 * v[...,0]
        v[...,0] = v[...,1] + v[...,2]
        return v[...,0]
