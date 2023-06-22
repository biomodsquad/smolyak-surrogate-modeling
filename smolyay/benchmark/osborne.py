import numpy

from .benchmark import BenchmarkFunction

class _osborne(BenchmarkFunction):
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,4))
        v[...,0] = x[...,0] + x[...,1]        
        v[...,0] += x[...,2]        
        v[...,1] = 0.844 - v[...,0]        
        v[...,0] = v[...,1] * v[...,1]        
        v[...,1] = 10. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 10. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.908 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 20. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 20. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.932 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 30. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 30. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.936 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 40. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 40. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.925 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 50. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 50. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.908 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 60. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 60. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.881 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 70. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 70. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.85 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 80. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 80. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.818 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 90. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 90. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.784 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 100. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 100. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.751 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 110. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 110. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.718 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 120. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 120. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.685 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 130. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 130. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.658 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 140. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 140. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.628 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 150. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 150. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.603 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 160. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 160. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.58 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 170. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 170. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.558 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 180. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 180. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.538 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 190. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 190. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.522 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 200. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 200. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.506 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 210. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 210. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.49 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 220. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 220. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.478 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 230. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 230. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.467 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 240. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 240. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.457 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 250. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 250. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.448 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 260. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 260. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.438 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 270. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 270. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.431 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 280. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 280. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.424 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 290. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 290. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.42 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]        
        v[...,3] = 300. * x[...,3]        
        v[...,1] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,1])        
        v[...,1] = x[...,1] * v[...,3]        
        v[...,1] += x[...,0]        
        v[...,3] = 300. * x[...,4]        
        v[...,2] = -v[...,3]        
        v[...,3] = numpy.exp(v[...,2])        
        v[...,2] = x[...,2] * v[...,3]        
        v[...,1] += v[...,2]        
        v[...,2] = 0.414 - v[...,1]        
        v[...,1] = v[...,2] * v[...,2]        
        v[...,0] += v[...,1]        
        v[...,1] = 310. * x[...,3]        
        v[...,2] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,2])        
        v[...,2] = x[...,1] * v[...,1]        
        v[...,2] += x[...,0]        
        v[...,1] = 310. * x[...,4]        
        v[...,3] = -v[...,1]        
        v[...,1] = numpy.exp(v[...,3])        
        v[...,3] = x[...,2] * v[...,1]        
        v[...,2] += v[...,3]        
        v[...,3] = 0.411 - v[...,2]        
        v[...,2] = v[...,3] * v[...,3]        
        v[...,0] += v[...,2]        
        v[...,2] = 320. * x[...,3]        
        v[...,3] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,3])        
        v[...,3] = x[...,1] * v[...,2]        
        v[...,3] += x[...,0]        
        v[...,2] = 320. * x[...,4]        
        v[...,1] = -v[...,2]        
        v[...,2] = numpy.exp(v[...,1])        
        v[...,1] = x[...,2] * v[...,2]        
        v[...,3] += v[...,1]        
        v[...,1] = 0.406 - v[...,3]        
        v[...,3] = v[...,1] * v[...,1]        
        v[...,0] += v[...,3]
        return v[...,0]


class osborne1(_osborne):
    @property
    def domain(self):
        # originally -2 and 2 for all, changed to avoid runtime warning
        return [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                [-1.0, 2.0], [-0.5, 0.5]]


class osbornea(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0],
                [-1.0, 1.0], [-1.0, 1.0]]
    def _function(self,x):
        v = numpy.zeros((x[...,0].size,4))
        v[...,0] = 0.844 - x[...,0]
        v[...,1] = v[...,0] - x[...,1]
        v[...,0] = v[...,1] - x[...,2]
        v[...,1] = v[...,0] * v[...,0]
        v[...,0] = 0.908 - x[...,0]
        v[...,2] = 10. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 10. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.932 - x[...,0]
        v[...,0] = 20. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 20. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.936 - x[...,0]
        v[...,2] = 30. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 30. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.925 - x[...,0]
        v[...,0] = 40. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 40. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.908 - x[...,0]
        v[...,2] = 50. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 50. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.881 - x[...,0]
        v[...,0] = 60. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 60. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.85 - x[...,0]
        v[...,2] = 70. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 70. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.818 - x[...,0]
        v[...,0] = 80. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 80. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.784 - x[...,0]
        v[...,2] = 90. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 90. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.751 - x[...,0]
        v[...,0] = 100. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 100. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.718 - x[...,0]
        v[...,2] = 110. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 110. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.685 - x[...,0]
        v[...,0] = 120. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 120. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.658 - x[...,0]
        v[...,2] = 130. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 130. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.628 - x[...,0]
        v[...,0] = 140. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 140. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.603 - x[...,0]
        v[...,2] = 150. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 150. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.58 - x[...,0]
        v[...,0] = 160. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 160. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.558 - x[...,0]
        v[...,2] = 170. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 170. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.538 - x[...,0]
        v[...,0] = 180. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 180. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.522 - x[...,0]
        v[...,2] = 190. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 190. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.506 - x[...,0]
        v[...,0] = 200. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 200. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.49 - x[...,0]
        v[...,2] = 210. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 210. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.478 - x[...,0]
        v[...,0] = 220. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 220. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.467 - x[...,0]
        v[...,2] = 230. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 230. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.457 - x[...,0]
        v[...,0] = 240. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 240. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.448 - x[...,0]
        v[...,2] = 250. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 250. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.438 - x[...,0]
        v[...,0] = 260. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 260. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.431 - x[...,0]
        v[...,2] = 270. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 270. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.424 - x[...,0]
        v[...,0] = 280. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 280. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.42 - x[...,0]
        v[...,2] = 290. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 290. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.414 - x[...,0]
        v[...,0] = 300. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 300. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        v[...,0] = 0.411 - x[...,0]
        v[...,2] = 310. * x[...,3]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = 310. * x[...,4]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = v[...,0] * v[...,0]
        v[...,1] += v[...,2]
        v[...,2] = 0.406 - x[...,0]
        v[...,0] = 320. * x[...,3]
        v[...,3] = -v[...,0]
        v[...,0] = numpy.exp(v[...,3])
        v[...,3] = x[...,1] * v[...,0]
        v[...,0] = v[...,2] - v[...,3]
        v[...,2] = 320. * x[...,4]
        v[...,3] = -v[...,2]
        v[...,2] = numpy.exp(v[...,3])
        v[...,3] = x[...,2] * v[...,2]
        v[...,2] = v[...,0] - v[...,3]
        v[...,0] = v[...,2] * v[...,2]
        v[...,1] += v[...,0]
        return v[...,1]