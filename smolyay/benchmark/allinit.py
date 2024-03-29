import numpy

from .benchmark import BenchmarkFunction

class allinit(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.1426691153, 8.8573308847], [1, 11.2456257795],[-1e10, 1]]
        
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [5])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,0] += v[...,1]
        v[...,1] = 2. + x[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = numpy.sin(x[...,2])
        v[...,1] = v[...,2] * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = x[...,0] * x[...,0]
        v[...,2] = x[...,1] * x[...,1]
        v[...,3] = v[...,1] * v[...,2]
        v[...,0] += v[...,3]
        v[...,3] = numpy.sin(x[...,2])
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = x[...,1] * x[...,1]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = x[...,2] * x[...,2]
        v[...,1] = 2. + x[...,0]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = v[...,3] + v[...,2]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = -3.173178189568194 + x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,2] = x[...,2] * x[...,2]
        v[...,4] = v[...,1] * v[...,2]
        v[...,3] += v[...,4]
        v[...,4] = v[...,3] * v[...,3]
        v[...,0] += v[...,4]
        v[...,0] += -0.3163656937942707
        rv = v[...,0] + x[...,2]
        return rv

class allinitu(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.5401384356, 10.31387540796], [-10.0, 9.0],
                [-9.9191099435, 9.07280105085],
                [-10.8111130846, 8.26999822386]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [5])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,0] += v[...,1]
        v[...,1] = x[...,2] + x[...,3]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = numpy.sin(x[...,2])
        v[...,1] = v[...,2] * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = x[...,0] * x[...,0]
        v[...,2] = x[...,1] * x[...,1]
        v[...,3] = v[...,1] * v[...,2]
        v[...,0] += v[...,3]
        v[...,3] = numpy.sin(x[...,2])
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -1. + x[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = x[...,1] * x[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = x[...,2] * x[...,2]
        v[...,3] = x[...,3] + x[...,0]
        v[...,2] = v[...,3] * v[...,3]
        v[...,3] = v[...,1] + v[...,2]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -4. + x[...,0]
        v[...,3] = numpy.sin(x[...,3])
        v[...,2] = v[...,3] * v[...,3]
        v[...,1] += v[...,2]
        v[...,2] = x[...,1] * x[...,1]
        v[...,3] = x[...,2] * x[...,2]
        v[...,4] = v[...,2] * v[...,3]
        v[...,1] += v[...,4]
        v[...,4] = v[...,1] * v[...,1]
        v[...,0] += v[...,4]
        v[...,4] = numpy.sin(x[...,3])
        v[...,1] = pow(v[...,4], 4)
        v[...,0] += v[...,1]
        v[...,0] += -4.
        rv = v[...,0] + x[...,2]
        rv += x[...,3]
        return rv
