import numpy

from .benchmark import BenchmarkFunction

class humps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0000000029, 8.99999999739], [-10.000000004, 8.9999999964]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 0.05 * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = 0.05 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 20. * x[...,0]
        v[...,0] = numpy.sin(v[...,2])
        v[...,2] = 20. * x[...,1]
        v[...,3] = numpy.sin(v[...,2])
        v[...,2] = v[...,0] * v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]
