import numpy

from .benchmark import BenchmarkFunction

class chi(BenchmarkFunction):
    @property
    def domain(self):
        return [[-30, 30], [-30, 30]]
        
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 1.5707963267948966 * x[...,0]
        v[...,2] = numpy.cos(v[...,1])
        v[...,1] = 10. * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = 15.707963267948966 * x[...,0]
        v[...,2] = numpy.sin(v[...,1])
        v[...,1] = 8. * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = -0.5 + x[...,1]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = v[...,2] / 2.
        v[...,2] = -v[...,1]
        v[...,1] = numpy.exp(v[...,2])
        v[...,2] = -0.4472135954999579 * v[...,1]
        v[...,0] += v[...,2]
        v[...,0] += 11.
        rv = v[...,0] + -12.*x[...,0]
        return rv
