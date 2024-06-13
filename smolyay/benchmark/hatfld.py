import numpy

from .benchmark import BenchmarkFunction

class _hatfld(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = -1. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = numpy.sqrt(x[...,1])
        v[...,2] = x[...,0] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = numpy.sqrt(x[...,2])
        v[...,2] = x[...,1] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = numpy.sqrt(x[...,3])
        v[...,2] = x[...,2] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]

class hatflda(_hatfld):
    @property
    def domain(self):
        return [[1e-07, 10.999999997], [1e-07, 10.9999999714],
                [1e-07, 10.9999999281], [1e-07, 10.9999998559]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [1, 1, 1, 1]

class hatfldb(_hatfld):
    @property
    def domain(self):
        return [[1e-07, 10.9472135922], [1e-07, 0.8],
                [1e-07, 10.6400000036], [1e-07, 10.4096000079]]

    @property
    def global_minimum(self):
        return 0.005572809
    
    @property
    def global_minimum_location(self):
        return [0.9472135922, 0.8, 0.6400000036, 0.4096000079]

class hatfldc(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10], [-8.9999999978, 11.0000000022]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [1, 1, 1, 1]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = -1. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = x[...,1] * x[...,1]
        v[...,2] = x[...,2] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = x[...,2] * x[...,2]
        v[...,2] = x[...,3] - v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = -1. + x[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        return v[...,1]

class hatfldd(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.8005223172, 11.87952991452],
                [-11.0178316754, 8.08395149214],
                [-10.7584301644, 8.31741285204]]

    @property
    def global_minimum(self):
        return 6.62e-08
    
    @property
    def global_minimum_location(self):
        return [3.1994776828, -1.0178316754, -0.7584301644]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 0.2 * x[...,2]
        v[...,1] = numpy.exp(v[...,0])
        v[...,0] = 0.2 * x[...,1]
        v[...,2] = numpy.exp(v[...,0])
        v[...,0] = x[...,0] * v[...,2]
        v[...,2] = v[...,1] - v[...,0]
        v[...,1] = 1.751 + v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = 0.3 * x[...,2]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.3 * x[...,1]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = x[...,0] * v[...,3]
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = 1.561 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.4 * x[...,2]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = 0.4 * x[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = v[...,0] - v[...,3]
        v[...,0] = 1.391 + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.5 * x[...,2]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.5 * x[...,1]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = x[...,0] * v[...,3]
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = 1.239 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.6 * x[...,2]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = 0.6 * x[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = v[...,0] - v[...,3]
        v[...,0] = 1.103 + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.7 * x[...,2]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.7 * x[...,1]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = x[...,0] * v[...,3]
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = 0.981 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.75 * x[...,2]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = 0.75 * x[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = v[...,0] - v[...,3]
        v[...,0] = 0.925 + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.8 * x[...,2]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.8 * x[...,1]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = x[...,0] * v[...,3]
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = 0.8721 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.85 * x[...,2]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = 0.85 * x[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = v[...,0] - v[...,3]
        v[...,0] = 0.8221 + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.9 * x[...,2]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.9 * x[...,1]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = x[...,0] * v[...,3]
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = 0.7748 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        return v[...,2]