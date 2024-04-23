import numpy

from .benchmark import BenchmarkFunction

class schwefel(BenchmarkFunction):
    @property
    def domain(self):
        return [[-0.5, 0.36], [-0.5, 0.36], [-0.5, 0.36],
                [-0.5, 0.36], [-0.5, 0.36]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [ 0.0714072734, 0.0714072734, 0.0714073339, 0.0714073339, 0.0714073339]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [2])
        v[...,0] = pow(x[...,0], 10)
        v[...,1] = pow(x[...,1], 10)
        v[...,0] += v[...,1]
        v[...,1] = pow(x[...,2], 10)
        v[...,0] += v[...,1]
        v[...,1] = pow(x[...,3], 10)
        v[...,0] += v[...,1]
        v[...,1] = pow(x[...,4], 10)
        v[...,0] += v[...,1]
        return v[...,0]
