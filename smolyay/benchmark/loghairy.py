import numpy

from .benchmark import BenchmarkFunction

class loghairy(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9999999999, 9.00000000009], [-9.9999999974, 9.00000000234]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = 7. * x[...,0]
        v[...,1] = numpy.sin(v[...,0])
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 7. * x[...,1]
        v[...,2] = numpy.cos(v[...,1])
        v[...,1] = v[...,2] * v[...,2]
        v[...,2] = v[...,0] * v[...,1]
        v[...,0] = 30. * v[...,2]
        v[...,0] += 100.
        v[...,2] = x[...,0] - x[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,2] = 0.01 + v[...,1]
        v[...,1] = numpy.sqrt(v[...,2])
        v[...,2] = 100. * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = x[...,0] * x[...,0]
        v[...,1] = 0.01 + v[...,2]
        v[...,2] = numpy.sqrt(v[...,1])
        v[...,1] = 100. * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = v[...,0] / 100.
        v[...,0] = numpy.log(v[...,1])
        return v[...,0]
