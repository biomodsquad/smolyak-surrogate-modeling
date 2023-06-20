import numpy

from .benchmark import BenchmarkFunction

class _camel(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 4. * v[...,0]
        v[...,0] = pow(x[...,0], 4.)
        v[...,2] = -2.1 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = pow(x[...,0], 6.)
        v[...,0] = 0.3333333333333333 * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = x[...,0] * x[...,1]
        v[...,1] += v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = -4. * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = pow(x[...,1], 4.)
        v[...,0] = 4. * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]

class camel1(_camel):
    @property
    def domain(self):
        return [[-5, 5], [-5, 5]]

class camel6(_camel):
    @property
    def domain(self):
        return [[-3, 3], [-1.5, 1.5]]
