import numpy

from .benchmark import BenchmarkFunction

class cliff(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7, 11.7], [-6.8502133863, 11.83480795233]]
        
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = 0.01 * x[...,0]
        v[...,1] = -0.03 + v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = x[...,0] - x[...,1]
        v[...,2] = 20. * v[...,1]
        v[...,1] = numpy.exp(v[...,2])
        v[...,2] = v[...,0] + v[...,1]
        rv = v[...,2] + -x[...,0]
        rv += x[...,1]
        return rv
