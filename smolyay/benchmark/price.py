import numpy

from .benchmark import BenchmarkFunction

class price(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    @property
    def global_minimum(self):
        return 3e-10
    
    @property
    def global_minimum_location(self):
        return [-0.0042706636, 0.0263165412]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = pow(x[...,0], 3)
        v[...,1] = 2. * v[...,0]
        v[...,0] = v[...,1] * x[...,1]
        v[...,1] = pow(x[...,1], 3)
        v[...,2] = v[...,0] - v[...,1]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 6. * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,3] = v[...,2] - v[...,1]
        v[...,2] = v[...,3] + x[...,1]
        v[...,3] = v[...,2] * v[...,2]
        v[...,2] = v[...,0] + v[...,3]
        return v[...,2]
