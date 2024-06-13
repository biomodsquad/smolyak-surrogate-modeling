import numpy

from .benchmark import BenchmarkFunction

class st_cqpjk2(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 0.9], [0, 0.9], [0, 0.9]]

    @property
    def global_minimum(self):
        return -12.5
    
    @property
    def global_minimum_location(self):
        return [0.8333333333, 0.6666666667, 0.5]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = 9. * x[...,0]
        v[...,1] = v[...,0] * x[...,0]
        v[...,0] = 9. * x[...,1]
        v[...,2] = v[...,0] * x[...,1]
        v[...,1] += v[...,2]
        v[...,2] = 9. * x[...,2]
        v[...,0] = v[...,2] * x[...,2]
        v[...,1] += v[...,0]
        rv = v[...,1] + -15.*x[...,0]
        rv += -12.*x[...,1]
        rv += -9.*x[...,2]
        return rv

class st_bsj3(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 99], [0, 99], [0, 99], [0, 99], [0, 99], [0, 99]]

    @property
    def global_minimum(self):
        return -86768.55
    
    @property
    def global_minimum_location(self):
        return [99, 99, 99, 99, 99, 99]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = -1.5 * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = -v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = x[...,2] * x[...,2]
        v[...,0] = -v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = x[...,3] * x[...,3]
        v[...,2] = -2. * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = x[...,4] * x[...,4]
        v[...,0] = -v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = x[...,5] * x[...,5]
        v[...,2] = -2.5 * v[...,0]
        v[...,1] += v[...,2]
        rv = v[...,1] + 10.5*x[...,0]
        rv += -3.95*x[...,1]
        rv += 3.*x[...,2]
        rv += 5.*x[...,3]
        rv += 1.5*x[...,4]
        rv += -1.5*x[...,5]
        return rv
