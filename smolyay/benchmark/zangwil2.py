import numpy

from .benchmark import BenchmarkFunction

class zangwil2(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.0, 12.6], [-1.0, 17.1]]

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 1.0666666666666667 * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = 1.0666666666666667 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 8. * x[...,0]
        v[...,0] = v[...,2] * x[...,1]
        v[...,2] = -0.06666666666666667 * v[...,0]
        v[...,1] += v[...,2]
        v[...,1] += 66.06666666666666
        rv = v[...,1] + -3.7333333333333334*x[...,0]
        rv += -17.066666666666666*x[...,1]
        return rv
