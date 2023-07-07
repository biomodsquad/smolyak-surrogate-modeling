import numpy

from .benchmark import BenchmarkFunction

class tre(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 5.0], [-5.0, 5.0]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = pow(x[...,0], 4)
        v[...,1] = pow(x[...,0], 3)
        v[...,2] = 4. * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = x[...,0] * x[...,0]
        v[...,1] = 4. * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = x[...,1] * x[...,1]
        v[...,0] += v[...,1]
        return v[...,0]
