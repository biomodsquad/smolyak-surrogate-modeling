import numpy

from .benchmark import BenchmarkFunction

class _beale(BenchmarkFunction):

    def _function(self,x):
        v = numpy.zeros((x[...,0].size,3))
        v[...,0] = 1. - x[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = -1.5 + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = 1. - v[...,0]
        v[...,0] = x[...,0] * v[...,2]
        v[...,2] = -2.25 + v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = pow(x[...,1], 3)
        v[...,2] = 1. - v[...,0]
        v[...,0] = x[...,0] * v[...,2]
        v[...,2] = -2.625 + v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]


class beale(_beale):
    @property
    def domain(self):
        return ([[-7.0000000008, 11.69999999928],
                          [-9.5000000002,9.44999999982]])
