import numpy

from .benchmark import BenchmarkFunction
from .camel import _camel

class ex4_1_5(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 10.0], [-10.0, 0.0]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 2. * v[...,0]
        v[...,0] = pow(x[...,0], 4)
        v[...,2] = -1.05 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = pow(x[...,0], 6)
        v[...,0] = 0.166666666666667 * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = x[...,0] * x[...,1]
        v[...,2] = -v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = x[...,1] * x[...,1]
        v[...,1] += v[...,2]

        return v[...,1]


class ex8_1_1(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.0, 2.0], [-1.0, 1.0]]

    @property
    def global_minimum(self):
        return -2.0218067834
    
    @property
    def global_minimum_location(self):
        return [ 2, 0.1057834695 ]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = numpy.cos(x[...,0])
        v[...,1] = numpy.sin(x[...,1])
        v[...,2] = v[...,0] * v[...,1]
        v[...,0] = x[...,1] * x[...,1]
        v[...,1] = 1. + v[...,0]
        v[...,0] = x[...,0] / v[...,1]
        v[...,1] = v[...,2] - v[...,0]

        return v[...,1]

class ex8_1_3(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.6, 8.46], [-10.4, 8.64]]

    @property
    def global_minimum(self):
        return 30
    
    @property
    def global_minimum_location(self):
        return [ -0.6, -0.4 ]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [5])
        v[...,0] = 1. + x[...,0]
        v[...,0] += x[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = x[...,0] * x[...,0]
        v[...,2] = 3. * v[...,0]
        v[...,2] += 19.
        v[...,0] = -14. * x[...,0]
        v[...,2] += v[...,0]
        v[...,0] = 6. * x[...,0]
        v[...,3] = v[...,0] * x[...,1]
        v[...,2] += v[...,3]
        v[...,3] = -14. * x[...,1]
        v[...,2] += v[...,3]
        v[...,3] = x[...,1] * x[...,1]
        v[...,0] = 3. * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = v[...,1] * v[...,2]
        v[...,1] = 1. + v[...,0]
        v[...,0] = 2. * x[...,0]
        v[...,2] = -3. * x[...,1]
        v[...,3] = v[...,0] + v[...,2]
        v[...,0] = v[...,3] * v[...,3]
        v[...,3] = x[...,0] * x[...,0]
        v[...,2] = 12. * v[...,3]
        v[...,2] += 18.
        v[...,3] = -32. * x[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 36. * x[...,0]
        v[...,4] = v[...,3] * x[...,1]
        v[...,3] = v[...,2] - v[...,4]
        v[...,2] = 48. * x[...,1]
        v[...,3] += v[...,2]
        v[...,2] = x[...,1] * x[...,1]
        v[...,4] = 27. * v[...,2]
        v[...,3] += v[...,4]
        v[...,4] = v[...,0] * v[...,3]
        v[...,0] = 30. + v[...,4]
        v[...,4] = v[...,1] * v[...,0]
        return v[...,4]


class ex8_1_4(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = 12. * v[...,0]
        v[...,0] = pow(x[...,0], 4)
        v[...,2] = -6.3 * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = pow(x[...,0], 6)
        v[...,1] += v[...,2]
        v[...,2] = 6. * x[...,0]
        v[...,0] = v[...,2] * x[...,1]
        v[...,2] = -v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = x[...,1] * x[...,1]
        v[...,0] = 6. * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]

class ex8_1_5(_camel):
    @property
    def domain(self):
        return [[-9.9101579868, 9.08085781188], [-10.7126564026, 8.35860923766]]
    
    @property
    def global_minimum(self):
        return -1.0316284535
    
    @property
    def global_minimum_location(self):
        return [ 0.0898420132, -0.7126564026 ]

class ex8_1_6(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.0000519964, 12.59995320324], [-6.0000519964, 12.59995320324]]

    @property
    def global_minimum(self):
        return -10.0860014962
    
    @property
    def global_minimum_location(self):
        return [ 3.9999480036, 3.9999480036 ]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = -4. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,1] += 0.1
        v[...,0] = -4. + x[...,1]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = -1. / v[...,1]
        v[...,1] = -1. + x[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,0] += 0.2
        v[...,1] = -1. + x[...,1]
        v[...,3] = v[...,1] * v[...,1]
        v[...,0] += v[...,3]
        v[...,3] = 1. / v[...,0]
        v[...,0] = -v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = -8. + x[...,0]
        v[...,3] = v[...,0] * v[...,0]
        v[...,3] += 0.2
        v[...,0] = -8. + x[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,3] += v[...,1]
        v[...,1] = 1. / v[...,3]
        v[...,3] = -v[...,1]
        v[...,2] += v[...,3]
        return v[...,2]
