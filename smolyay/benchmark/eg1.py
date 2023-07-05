import numpy

from .benchmark import BenchmarkFunction

class eg1(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.2302657121, 9.7697342879], [-1, 1], [1, 2]]
        
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,2]
        v[...,2] = pow(v[...,1], 4.)
        v[...,0] += v[...,2]
        v[...,2] = x[...,0] * x[...,2]
        v[...,0] += v[...,2]
        v[...,2] = x[...,0] + x[...,2]
        v[...,1] = numpy.sin(v[...,2])
        v[...,2] = x[...,1] * v[...,1]
        v[...,0] += v[...,2]
        rv = v[...,0] + x[...,1]
        return rv
