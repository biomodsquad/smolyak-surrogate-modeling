import numpy

from .benchmark import BenchmarkFunction


class box2(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]
        
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 0.1 * x[...,0]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.1 * x[...,1]
        v[...,2] = -v[...,1]
        v[...,1] = numpy.exp(v[...,2])
        v[...,2] = v[...,0] - v[...,1]
        v[...,0] = -0.5369579768645172 + v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = 0.2 * x[...,0]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.2 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = -0.6833954698413691 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.30000000000000004 * x[...,0]
        v[...,0] = -v[...,3]
        v[...,3] = numpy.exp(v[...,0])
        v[...,0] = 0.30000000000000004 * x[...,1]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = v[...,3] - v[...,0]
        v[...,3] = -0.6910311523138539 + v[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,2] += v[...,1]
        v[...,1] = 0.4 * x[...,0]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = 0.4 * x[...,1]
        v[...,0] = -v[...,3]
        v[...,3] = numpy.exp(v[...,0])
        v[...,0] = v[...,1] - v[...,3]
        v[...,1] = -0.6520044071469051 + v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,2] += v[...,0]
        v[...,0] = 0.5 * x[...,0]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.5 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = -0.599792712713548 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.6000000000000001 * x[...,0]
        v[...,0] = -v[...,3]
        v[...,3] = numpy.exp(v[...,0])
        v[...,0] = 0.6000000000000001 * x[...,1]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = v[...,3] - v[...,0]
        v[...,3] = -0.54633288391736 + v[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,2] += v[...,1]
        v[...,1] = 0.7000000000000001 * x[...,0]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = 0.7000000000000001 * x[...,1]
        v[...,0] = -v[...,3]
        v[...,3] = numpy.exp(v[...,0])
        v[...,0] = v[...,1] - v[...,3]
        v[...,1] = -0.49567342182585494 + v[...,0]
        v[...,0] = v[...,1] * v[...,1]
        v[...,2] += v[...,0]
        v[...,0] = 0.8 * x[...,0]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.8 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,0] - v[...,1]
        v[...,0] = -0.44899350148931905 + v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 0.9 * x[...,0]
        v[...,0] = -v[...,3]
        v[...,3] = numpy.exp(v[...,0])
        v[...,0] = 0.9 * x[...,1]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = v[...,3] - v[...,0]
        v[...,3] = -0.4064462499365124 + v[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,2] += v[...,1]
        v[...,1] = -x[...,0]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = -x[...,1]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = v[...,3] - v[...,0]
        v[...,3] = -0.36783404124167984 + v[...,1]
        v[...,1] = v[...,3] * v[...,3]
        v[...,2] += v[...,1]
        return v[...,2]

class _box(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 0.1 * x[...,0]
        v[...,1] = -v[...,0]
        v[...,0] = numpy.exp(v[...,1])
        v[...,1] = 0.1 * x[...,1]
        v[...,2] = -v[...,1]
        v[...,1] = numpy.exp(v[...,2])
        v[...,2] = v[...,0] - v[...,1]
        v[...,0] = -0.9048374180359595 * x[...,2]
        v[...,2] += v[...,0]
        v[...,0] = 0.36787944117144233 * x[...,2]
        v[...,2] += v[...,0]
        v[...,0] = v[...,2] * v[...,2]
        v[...,2] = 0.2 * x[...,0]
        v[...,1] = -v[...,2]
        v[...,2] = numpy.exp(v[...,1])
        v[...,1] = 0.2 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,2] - v[...,1]
        v[...,2] = -0.8187307530779818 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = 0.1353352832366127 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = v[...,3] * v[...,3]
        v[...,0] += v[...,2]
        v[...,2] = 0.30000000000000004 * x[...,0]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = 0.30000000000000004 * x[...,1]
        v[...,1] = -v[...,3]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = v[...,2] - v[...,3]
        v[...,2] = -0.7408182206817179 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = 0.04978706836786393 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = 0.4 * x[...,0]
        v[...,1] = -v[...,2]
        v[...,2] = numpy.exp(v[...,1])
        v[...,1] = 0.4 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,2] - v[...,1]
        v[...,2] = -0.6703200460356393 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = 0.018315638888734182 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = v[...,3] * v[...,3]
        v[...,0] += v[...,2]
        v[...,2] = 0.5 * x[...,0]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = 0.5 * x[...,1]
        v[...,1] = -v[...,3]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = v[...,2] - v[...,3]
        v[...,2] = -0.6065306597126334 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = 0.006737946999085465 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = 0.6000000000000001 * x[...,0]
        v[...,1] = -v[...,2]
        v[...,2] = numpy.exp(v[...,1])
        v[...,1] = 0.6000000000000001 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,2] - v[...,1]
        v[...,2] = -0.5488116360940264 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = 0.002478752176666357 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = v[...,3] * v[...,3]
        v[...,0] += v[...,2]
        v[...,2] = 0.7000000000000001 * x[...,0]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = 0.7000000000000001 * x[...,1]
        v[...,1] = -v[...,3]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = v[...,2] - v[...,3]
        v[...,2] = -0.49658530379140947 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = 0.0009118819655545155 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = 0.8 * x[...,0]
        v[...,1] = -v[...,2]
        v[...,2] = numpy.exp(v[...,1])
        v[...,1] = 0.8 * x[...,1]
        v[...,3] = -v[...,1]
        v[...,1] = numpy.exp(v[...,3])
        v[...,3] = v[...,2] - v[...,1]
        v[...,2] = -0.44932896411722156 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = 0.00033546262790251185 * x[...,2]
        v[...,3] += v[...,2]
        v[...,2] = v[...,3] * v[...,3]
        v[...,0] += v[...,2]
        v[...,2] = 0.9 * x[...,0]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = 0.9 * x[...,1]
        v[...,1] = -v[...,3]
        v[...,3] = numpy.exp(v[...,1])
        v[...,1] = v[...,2] - v[...,3]
        v[...,2] = -0.4065696597405991 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = 0.00012340980408667953 * x[...,2]
        v[...,1] += v[...,2]
        v[...,2] = v[...,1] * v[...,1]
        v[...,0] += v[...,2]
        v[...,2] = -x[...,0]
        v[...,1] = numpy.exp(v[...,2])
        v[...,2] = -x[...,1]
        v[...,3] = numpy.exp(v[...,2])
        v[...,2] = v[...,1] - v[...,3]
        v[...,1] = -0.36787944117144233 * x[...,2]
        v[...,2] += v[...,1]
        v[...,1] = 4.539992976248483e-05 * x[...,2]
        v[...,2] += v[...,1]
        v[...,1] = v[...,2] * v[...,2]
        v[...,0] += v[...,1]
        return v[...,0]
   
class box3(_box):
    @property
    def domain(self):
        return [[-9.0000004305, 9.89999961255],
                [3.23989999984065e-06, 18.00000291591],
                [-8.9999997323, 9.90000024093]]