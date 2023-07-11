import numpy

from .benchmark import BenchmarkFunction

class sim2bqp(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 9], [0, 0.45]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = -x[...,0]
        v[...,1] = v[...,0] + x[...,1]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = x[...,0] + x[...,1]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = v[...,0] + v[...,2]
        rv = v[...,1] + x[...,1]
        return rv

class simbqp(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 9], [0, 0.45]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = -x[...,0]
        v[...,1] = v[...,0] + x[...,1]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 2. * x[...,0]
        v[...,2] = v[...,1] + x[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,2] = v[...,0] + v[...,1]
        rv = v[...,2] + x[...,1]
        return rv
