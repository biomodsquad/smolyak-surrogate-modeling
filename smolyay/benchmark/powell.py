import numpy

from .benchmark import BenchmarkFunction

class _powell(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = 10. * x[...,1]
        v[...,1] = x[...,0] + v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = x[...,2] - x[...,3]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = 5. * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = -2. * x[...,2]
        v[...,2] = x[...,1] + v[...,1]
        v[...,1] = pow(v[...,2], 4)
        v[...,0] += v[...,1]
        v[...,1] = x[...,0] - x[...,3]
        v[...,2] = pow(v[...,1], 4)
        v[...,1] = 10. * v[...,2]
        v[...,0] += v[...,1]
        return v[...,0]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [2.9999999992, 0.4999999998]

class powell(_powell):
    @property
    def domain(self):
        return [[-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0.0015780539, -0.0001578054, 0.0008014333, 0.0008014333]