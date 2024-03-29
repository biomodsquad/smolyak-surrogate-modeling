import numpy

from .benchmark import BenchmarkFunction

class shekel(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10], [0, 10]]
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = -4. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,1] += 0.1
        v[...,0] = -4. + x[...,1]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = -4. + x[...,2]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = -4. + x[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 1. / v[...,1]
        v[...,1] = -1. + x[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,0] += 0.2
        v[...,1] = -1. + x[...,1]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = -1. + x[...,2]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -1. + x[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = 1. / v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = -8. + x[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,0] += 0.2
        v[...,3] = -8. + x[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -8. + x[...,2]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = -8. + x[...,3]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = 1. / v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = -6. + x[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,0] += 0.4
        v[...,1] = -6. + x[...,1]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = -6. + x[...,2]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -6. + x[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = 1. / v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = -3. + x[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,0] += 0.4
        v[...,3] = -7. + x[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = -3. + x[...,2]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = -7. + x[...,3]
        v[...,1] = v[...,3] * v[...,3]
        v[...,0] += v[...,1]
        v[...,1] = 1. / v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = -v[...,2]
        return v[...,1]
