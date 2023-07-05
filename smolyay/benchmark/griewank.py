import numpy

from .benchmark import BenchmarkFunction

class griewank(BenchmarkFunction):
    @property
    def domain(self):
        return [[-100.0, 90.0], [-100.0, 90.0]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 0.005 * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = 0.005 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = numpy.cos(x[...,0])
        v[...,0] = x[...,1] / 1.4142135623730951
        v[...,3] = numpy.cos(v[...,0])
        v[...,0] = v[...,2] * v[...,3]
        v[...,2] = -v[...,0]
        v[...,1] += v[...,2]
        v[...,1] += 1.
        return v[...,1]
