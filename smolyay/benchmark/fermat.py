import numpy

from .benchmark import BenchmarkFunction

class fermat_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.9999999999, 12.0000000001],
                [-8.8452994616, 11.1547005384], [1e-08, 10.00000001]]

    @property
    def global_minimum(self):
        return 7.4641016251
    
    @property
    def global_minimum_location(self):
        return [2.0000000001, 1.1547005384, 1e-08]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [5])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,2] = x[...,2] * x[...,2]
        v[...,3] = v[...,1] + v[...,2]
        v[...,1] = v[...,0] + v[...,3]
        v[...,0] = numpy.sqrt(v[...,1])
        v[...,1] = x[...,0] * x[...,0]
        v[...,3] = -4. + x[...,1]
        v[...,2] = v[...,3] * v[...,3]
        v[...,3] = x[...,2] * x[...,2]
        v[...,4] = v[...,2] + v[...,3]
        v[...,2] = v[...,1] + v[...,4]
        v[...,1] = numpy.sqrt(v[...,2])
        v[...,0] += v[...,1]
        v[...,1] = x[...,0] * x[...,0]
        v[...,2] = -2. + x[...,1]
        v[...,4] = v[...,2] * v[...,2]
        v[...,2] = -4. + x[...,2]
        v[...,3] = v[...,2] * v[...,2]
        v[...,2] = v[...,4] + v[...,3]
        v[...,4] = v[...,1] + v[...,2]
        v[...,1] = numpy.sqrt(v[...,4])
        v[...,0] += v[...,1]
        rv = v[...,0] + x[...,0]
        return rv

class fermat2_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8, 12], [-9.00000002, 10.99999998], [1e-08, 10.00000001]]

    @property
    def global_minimum(self):
        return 4.4721359695
    
    @property
    def global_minimum_location(self):
        return  [2, 0.99999998, 1e-08]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [5])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,2] = x[...,2] * x[...,2]
        v[...,3] = v[...,1] + v[...,2]
        v[...,1] = v[...,0] + v[...,3]
        v[...,0] = numpy.sqrt(v[...,1])
        v[...,1] = x[...,0] * x[...,0]
        v[...,3] = -4. + x[...,1]
        v[...,2] = v[...,3] * v[...,3]
        v[...,3] = x[...,2] * x[...,2]
        v[...,4] = v[...,2] + v[...,3]
        v[...,2] = v[...,1] + v[...,4]
        v[...,1] = numpy.sqrt(v[...,2])
        v[...,0] += v[...,1]
        v[...,1] = x[...,0] * x[...,0]
        v[...,2] = -2. + x[...,1]
        v[...,4] = v[...,2] * v[...,2]
        v[...,2] = -1. + x[...,2]
        v[...,3] = v[...,2] * v[...,2]
        v[...,2] = v[...,4] + v[...,3]
        v[...,4] = v[...,1] + v[...,2]
        v[...,1] = numpy.sqrt(v[...,4])
        v[...,0] += v[...,1]
        rv = v[...,0] + x[...,0]
        return rv
