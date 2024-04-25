import numpy

from .benchmark import BenchmarkFunction

class denschna(BenchmarkFunction):
    @property
    def domain(self):
        return [[-20, 9], [-20, 9]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = pow(x[...,0], 4.)
        v[...,1] = x[...,0] + x[...,1]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = numpy.exp(x[...,1])
        v[...,1] = -1. + v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        return v[...,0]

class denschnb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.0, 10.8], [-11.0, 8.1]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [2, -1]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = -2. + x[...,0]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = -2. + x[...,0]
        v[...,2] = v[...,0] * x[...,1]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 1. + x[...,1]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        return v[...,1]

class denschnc(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [1, 1]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,0] += -2.
        v[...,1] = x[...,1] * x[...,1]
        v[...,0] += v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = -1. + x[...,0]
        v[...,2] = numpy.exp(v[...,0])
        v[...,2] += -2.
        v[...,0] = pow(x[...,1], 3)
        v[...,2] += v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = v[...,1] + v[...,0]
        return v[...,2]
    

class denschnd(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0002179404, 8.99980385364], [-9.9733994128, 9.02394052848], [-10.0001458391, 8.99986874481]]

    @property
    def global_minimum(self):
        return 4e-10
    
    @property
    def global_minimum_location(self):
        return [ -0.0002179404, 0.0266005872, -0.0001458391 ]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = pow(x[...,1], 3)
        v[...,2] = v[...,0] + v[...,1]
        v[...,0] = pow(x[...,2], 4)
        v[...,1] = v[...,2] - v[...,0]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = 2. * x[...,0]
        v[...,0] = v[...,1] * x[...,1]
        v[...,1] = v[...,0] * x[...,2]
        v[...,0] = v[...,1] * v[...,1]
        v[...,2] += v[...,0]
        v[...,0] = 2. * x[...,0]
        v[...,1] = v[...,0] * x[...,1]
        v[...,0] = 3. * x[...,1]
        v[...,3] = v[...,0] * x[...,2]
        v[...,0] = v[...,1] - v[...,3]
        v[...,1] = x[...,0] * x[...,2]
        v[...,3] = v[...,0] + v[...,1]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]

        return v[...,2]

class denschne(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0], [-10.0, 9.0]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [0, 0, 0]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [3])
        v[...,0] = x[...,0] * x[...,0]
        v[...,1] = x[...,1] * x[...,1]
        v[...,2] = x[...,1] + v[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,0] += v[...,1]
        v[...,1] = numpy.exp(x[...,2])
        v[...,2] = -1. + v[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,0] += v[...,1]
        return v[...,0]

class denschnf(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10000.0, 10000.0], [-10000.0, 10000.0]]

    @property
    def global_minimum(self):
        return 0
    
    @property
    def global_minimum_location(self):
        return [1, 1]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = x[...,0] + x[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = 2. * v[...,1]
        v[...,1] = x[...,0] - x[...,1]
        v[...,2] = v[...,1] * v[...,1]
        v[...,1] = v[...,0] + v[...,2]
        v[...,0] = -8. + v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = x[...,0] * x[...,0]
        v[...,2] = 5. * v[...,0]
        v[...,0] = -3. + x[...,1]
        v[...,3] = v[...,0] * v[...,0]
        v[...,0] = v[...,2] + v[...,3]
        v[...,2] = -9. + v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = v[...,1] + v[...,0]
        return v[...,2]