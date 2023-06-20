import numpy

from .benchmark import BenchmarkFunction

class prob09(BenchmarkFunction):
    @property
    def domain(self):
        return [[-2,2],[-2,2]]

    def _function(self,x):
        # does not have a C file to go with it
        return 100*(x[...,1]-(x[...,0])**2)**2+(1-x[...,0])**2
