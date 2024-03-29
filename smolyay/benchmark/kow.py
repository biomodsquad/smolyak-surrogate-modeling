import numpy

from .benchmark import BenchmarkFunction

class kowalik(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 0.378], [0, 0.378], [0, 0.378], [0, 0.378]]

    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 4. * x[...,1]
        v[...,1] = 16. + v[...,0]
        v[...,0] = x[...,0] * v[...,1]
        v[...,1] = 4. * x[...,2]
        v[...,1] += 16.
        v[...,1] += x[...,3]
        v[...,2] = v[...,0] / v[...,1]
        v[...,0] = 0.1957 - v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = 2. * x[...,1]
        v[...,1] = 4. + v[...,0]
        v[...,0] = x[...,0] * v[...,1]
        v[...,1] = 2. * x[...,2]
        v[...,1] += 4.
        v[...,1] += x[...,3]
        v[...,3] = v[...,0] / v[...,1]
        v[...,0] = 0.1947 - v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 1. + x[...,1]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 1. + x[...,2]
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.1735 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.5 * x[...,1]
        v[...,0] = 0.25 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.5 * x[...,2]
        v[...,0] += 0.25
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.16 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.25 * x[...,1]
        v[...,1] = 0.0625 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.25 * x[...,2]
        v[...,1] += 0.0625
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0844 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = 0.16666666666666666 * x[...,1]
        v[...,3] = 0.027777777777777776 + v[...,0]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 0.16666666666666666 * x[...,2]
        v[...,3] += 0.027777777777777776
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.0627 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.125 * x[...,1]
        v[...,0] = 0.015625 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.125 * x[...,2]
        v[...,0] += 0.015625
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.0456 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.1 * x[...,1]
        v[...,1] = 0.010000000000000002 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.1 * x[...,2]
        v[...,1] += 0.010000000000000002
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0342 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = 0.08333333333333333 * x[...,1]
        v[...,3] = 0.006944444444444444 + v[...,0]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 0.08333333333333333 * x[...,2]
        v[...,3] += 0.006944444444444444
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.0323 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.07142857142857142 * x[...,1]
        v[...,0] = 0.00510204081632653 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.07142857142857142 * x[...,2]
        v[...,0] += 0.00510204081632653
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.0235 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.0625 * x[...,1]
        v[...,1] = 0.00390625 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.0625 * x[...,2]
        v[...,1] += 0.00390625
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0246 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        return v[...,2]

class _kow(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros(list(x.shape[:-1]) + [4])
        v[...,0] = 4. * x[...,1]
        v[...,1] = 16. + v[...,0]
        v[...,0] = x[...,0] * v[...,1]
        v[...,1] = 4. * x[...,2]
        v[...,1] += 16.
        v[...,1] += x[...,3]
        v[...,2] = v[...,0] / v[...,1]
        v[...,0] = 0.1957 - v[...,2]
        v[...,2] = v[...,0] * v[...,0]
        v[...,0] = 2. * x[...,1]
        v[...,1] = 4. + v[...,0]
        v[...,0] = x[...,0] * v[...,1]
        v[...,1] = 2. * x[...,2]
        v[...,1] += 4.
        v[...,1] += x[...,3]
        v[...,3] = v[...,0] / v[...,1]
        v[...,0] = 0.1947 - v[...,3]
        v[...,3] = v[...,0] * v[...,0]
        v[...,2] += v[...,3]
        v[...,3] = 1. + x[...,1]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 1. + x[...,2]
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.1735 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.5 * x[...,1]
        v[...,0] = 0.25 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.5 * x[...,2]
        v[...,0] += 0.25
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.16 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.25 * x[...,1]
        v[...,1] = 0.0625 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.25 * x[...,2]
        v[...,1] += 0.0625
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0844 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = 0.167 * x[...,1]
        v[...,3] = 0.027889000000000004 + v[...,0]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 0.167 * x[...,2]
        v[...,3] += 0.027889000000000004
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.0627 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.125 * x[...,1]
        v[...,0] = 0.015625 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.125 * x[...,2]
        v[...,0] += 0.015625
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.0456 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.1 * x[...,1]
        v[...,1] = 0.010000000000000002 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.1 * x[...,2]
        v[...,1] += 0.010000000000000002
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0342 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        v[...,0] = 0.0833 * x[...,1]
        v[...,3] = 0.00693889 + v[...,0]
        v[...,0] = x[...,0] * v[...,3]
        v[...,3] = 0.0833 * x[...,2]
        v[...,3] += 0.00693889
        v[...,3] += x[...,3]
        v[...,1] = v[...,0] / v[...,3]
        v[...,0] = 0.0323 - v[...,1]
        v[...,1] = v[...,0] * v[...,0]
        v[...,2] += v[...,1]
        v[...,1] = 0.0714 * x[...,1]
        v[...,0] = 0.00509796 + v[...,1]
        v[...,1] = x[...,0] * v[...,0]
        v[...,0] = 0.0714 * x[...,2]
        v[...,0] += 0.00509796
        v[...,0] += x[...,3]
        v[...,3] = v[...,1] / v[...,0]
        v[...,1] = 0.0235 - v[...,3]
        v[...,3] = v[...,1] * v[...,1]
        v[...,2] += v[...,3]
        v[...,3] = 0.0625 * x[...,1]
        v[...,1] = 0.00390625 + v[...,3]
        v[...,3] = x[...,0] * v[...,1]
        v[...,1] = 0.0625 * x[...,2]
        v[...,1] += 0.00390625
        v[...,1] += x[...,3]
        v[...,0] = v[...,3] / v[...,1]
        v[...,3] = 0.0246 - v[...,0]
        v[...,0] = v[...,3] * v[...,3]
        v[...,2] += v[...,0]
        return v[...,2]

class kowosb(_kow):
    @property
    def domain(self):
        return [[-9.8071930634, 9.17352624294], [-9.8087176971, 9.17215407261],
                [-9.8769435657, 9.11075079087], [-9.8639376421, 9.12245612211]]
