import numpy

from .benchmark import BenchmarkFunction

class genhumps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.999999999, 9.0000000009], [-10.0000000017, 8.99999999847], [-10.0000000095, 8.99999999145], [-9.9999999989, 9.00000000099], [-10.0000000027, 8.99999999757]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0, 0, 0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 2. * x[...,0]
        v[...,1] = numpy.sin(v[...,0])
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 2. * x[...,1]
        v[...,2] = numpy.sin(v[...,1])
        v[...,1] = v[...,2] * v[...,2]
        v[...,2] = v[...,0] * v[...,1]
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 0.05 * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = x[...,1] * x[...,1]
        v[...,0] = 0.05 * v[...,1]
        v[...,2] += v[...,0]
        v[...,0] = 2. * x[...,1]
        v[...,1] = numpy.sin(v[...,0])
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = 2. * x[...,2]
        v[...,3] = numpy.sin(v[...,1])
        v[...,1] = v[...,3] * v[...,3]
        v[...,3] = v[...,0] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = x[...,1] * x[...,1]
        v[...,0] = 0.05 * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = x[...,2] * x[...,2]
        v[...,3] = 0.05 * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 2. * x[...,2]
        v[...,0] = numpy.sin(v[...,3])
        v[...,3] = v[...,0] * v[...,0]
        v[...,0] = 2. * x[...,3]
        v[...,1] = numpy.sin(v[...,0])
        v[...,0] = v[...,1] * v[...,1]
        v[...,1] = v[...,3] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = x[...,2] * x[...,2]
        v[...,3] = 0.05 * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = x[...,3] * x[...,3]
        v[...,1] = 0.05 * v[...,3]
        v[...,2] += v[...,1]
        v[...,1] = 2. * x[...,3]
        v[...,3] = numpy.sin(v[...,1])
        v[...,1] = v[...,3] * v[...,3]
        v[...,3] = 2. * x[...,4]
        v[...,0] = numpy.sin(v[...,3])
        v[...,3] = v[...,0] * v[...,0]
        v[...,0] = v[...,1] * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = x[...,3] * x[...,3]
        v[...,1] = 0.05 * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = x[...,4] * x[...,4]
        v[...,0] = 0.05 * v[...,1]
        v[...,2] += v[...,0]
        return v[...,2]
