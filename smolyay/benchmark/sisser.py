import numpy

from .benchmark import BenchmarkFunction

class sisser(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9978640372, 9.00192236652], [-9.9983980285, 9.00144177435]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0.0021359628, 0.0016019715]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = pow(x[...,0], 4)
        v[...,1] = 3. * v[...,0]
        v[...,0] = x[...,0] * x[...,1]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = -2. * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = pow(x[...,1], 4)
        v[...,2] = 3. * v[...,0]
        v[...,1] += v[...,2]
        return v[...,1]
