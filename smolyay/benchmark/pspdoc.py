import numpy

from .benchmark import BenchmarkFunction

class pspdoc(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.0, 0.0], [-9.999999972, 10.000000028], [-9.9999999213, 10.0000000787], [-9.9999998676, 10.0000001324]]

    @property
    def global_minimum(self):
        return 2.4142135624
    
    @property
    def global_minimum_location(self):
        return [-1, 2.8e-08, 7.87e-08, 1.324e-07]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] * x[...,0]
        v[...,0] += 1.
        v[...,1] = x[...,1] - x[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = numpy.sqrt(v[...,0])
        v[...,0] = x[...,1] * x[...,1]
        v[...,0] += 1.
        v[...,1] = x[...,2] - x[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = numpy.sqrt(v[...,0])
        v[...,0] = v[...,2] + v[...,3]
        return v[...,0]
