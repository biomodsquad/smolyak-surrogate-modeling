import abc

import numpy

class BenchmarkFunction(abc.ABC):
    """Benchmark Function

    Implementation of benchmark problems used for optimization, testing, and 
    analysis of surrogate function solvers. These benchmark problems come from
    https://sahinidis.coe.gatech.edu/dfo

    These functions operate on a defined `domain` they can be evaluated on,
    and the upper and lower bounds of this domain can be the domain in which
    solutions exist or can be arbitrary.

    """
    def __call__(self,x):
        """Evaluate the function.

        Parameters
        ----------
        x : list
            Function input.

        Raises
        ------
        ValueError
            If the input is outside the function domain.
        """
        if self.dimension > 1:
            oob = any(
                xi < bound[0] or xi > bound[1] for xi,bound in
                zip(x,self.domain))
        else:
            oob = x < self.domain[0][0] or x > self.domain[0][1]
        if oob:
            raise ValueError("Input out domain of function.")
        return self._function(x)
    
    @property
    def name(self):
        """Name of the function"""
        return type(self).__name__

    @property
    def dimension(self):
        """int: Number of variables."""
        return len(self.domain)

    @property
    def lower_bounds(self):
        """list: the lower bounds of the domain of each variable."""
        return [bound[0] for bound in self.domain]
    
    @property
    def upper_bounds(self):
        """list: the upper bounds of the domain of each variable."""
        return [bound[1] for bound in self.domain]

    @property
    @abc.abstractmethod
    def domain(self):
        """list: Domain of the function.
        
        The domain must be specified as lower and upper bounds for each variable as a list of lists.
        """
        pass
 
    @abc.abstractmethod
    def _function(self,x):
        pass

class beale(BenchmarkFunction):
    @property
    def domain(self):
        return ([[-7.0000000008, 11.69999999928],
                          [-9.5000000002,9.44999999982]])

    def _function(self,x):
        return ((-1.5 + x[0]*(1 - x[1]))**2 +
                (-2.25 + (1 - (x[1])**2)*x[0])**2 +
                (-2.625+(1-x[1]**3)*x[0])**2)

class box2(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]
        
    def _function(self,x):
        term1 = [0.536957976864517,0.683395469841369,0.691031152313854,
                 0.652004407146905,0.599792712713548,0.546332883917360,
                 0.495673421825855,0.448993501489319,0.406446249936512,
                 0.36783404124168]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        y = numpy.sum((numpy.exp(numpy.multiply(term2,x[0]))-
                       numpy.exp(numpy.multiply(term2,x[1]))-term1)**2)
        return y

class branin(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5, 10], [0, 15]]
        
    def _function(self,x):
        y1 = (x[1] - 5.1*x[0]**2/(4*numpy.pi**2) + 5*x[0]/numpy.pi - 6)**2
        y2 = 10*(1-1/(8*numpy.pi))*numpy.cos(x[0]) + 10
        return y1 + y2

# new
class brownbs(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10000000.0, 10000000.0], [-10000.0, 10000.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -1.e+06 + x[0]
        v[1] = v[0] * v[0]
        v[0] = -2.e-06 + x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[0] * x[1]
        v[0] = -2. + v[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        return v[1]


class _camel(BenchmarkFunction):
    def _function(self,x):
        return (4*x[0]**2-2.1*x[0]**4+0.333333333333333*x[0]**6 +
                x[0]*x[1]-4*x[1]**2+4*x[1]**4)

class camel1(_camel):
    @property
    def domain(self):
        return [[-5, 5], [-5, 5]]

class camel6(_camel):
    @property
    def domain(self):
        return [[-3, 3], [-1.5, 1.5]]

class chi(BenchmarkFunction):
    @property
    def domain(self):
        return [[-30, 30], [-30, 30]]
        
    def _function(self,x):
        return (x[0]**2 - 12*x[0] + 10*numpy.cos(1.5707963267949*x[0]) +
                8*numpy.sin(15.707963267949*x[0]) -
                0.447213595499958*numpy.exp(-0.5*(-0.5+x[1])**2) + 11)
    
class cliff(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7, 11.7], [-6.8502133863, 11.83480795233]]
        
    def _function(self,x):
        return (-0.03+0.01*x[0])**2-x[0]+numpy.exp(20*x[0]-20*x[1])+x[1]
    
class cube(BenchmarkFunction):
    @property
    def domain(self):
        return [[-18, 9.9], [-18, 9.9]]
        
    def _function(self,x):
        return (-1+x[0])**2+100*(x[1]-x[0]**3)**2
    
class denschna(BenchmarkFunction):
    @property
    def domain(self):
        return [[-20, 9], [-20, 9]]
        
    def _function(self,x):
        return x[0]**4+(x[0]+x[1])**2+(numpy.exp(x[1])-1)**2

# new
class denschnb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.0, 10.8], [-11.0, 8.1]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -2. + x[0]
        v[1] = v[0] * v[0]
        v[0] = -2. + x[0]
        v[2] = v[0] * x[1]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 1. + x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        return v[1]

# new
class denschnc(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[0] += -2.;
        v[1] = x[1] * x[1]
        v[0] += v[1]
        v[1] = v[0] * v[0]
        v[0] = -1. + x[0]
        v[2] = numpy.exp(v[0]);
        v[2] += -2.;
        v[0] = pow(x[1], 3)
        v[2] += v[0]
        v[0] = v[2] * v[2]
        v[2] = v[1] + v[0]
        return v[2]

# new
class denschnf(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10000.0, 10000.0], [-10000.0, 10000.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] + x[1]
        v[1] = v[0] * v[0]
        v[0] = 2. * v[1]
        v[1] = x[0] - x[1]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        v[0] = -8. + v[1]
        v[1] = v[0] * v[0]
        v[0] = x[0] * x[0]
        v[2] = 5. * v[0]
        v[0] = -3. + x[1]
        v[3] = v[0] * v[0]
        v[0] = v[2] + v[3]
        v[2] = -9. + v[0]
        v[0] = v[2] * v[2]
        v[2] = v[1] + v[0]

        return v[2]

# new
class ex4_1_5(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 10.0], [-10.0, 0.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = 2. * v[0]
        v[0] = pow(x[0], 4)
        v[2] = -1.05 * v[0]
        v[1] += v[2]
        v[2] = pow(x[0], 6)
        v[0] = 0.166666666666667 * v[2]
        v[1] += v[0]
        v[0] = x[0] * x[1]
        v[2] = -v[0]
        v[1] += v[2]
        v[2] = x[1] * x[1]
        v[1] += v[2]

        return v[1]

# new
class ex8_1_1(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.0, 2.0], [-1.0, 1.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = numpy.cos(x[0])
        v[1] = numpy.sin(x[1])
        v[2] = v[0] * v[1]
        v[0] = x[1] * x[1]
        v[1] = 1. + v[0]
        # here there was a check if v[1] was 0
        # this is the best guess on how that part of code is interpreted
        if not v[1] == 0:
            v[0] = x[0] / v[1]
        else:
            v[0] = x[0]
        v[1] = v[2] - v[0]

        return v[1]
# new
class ex8_1_3(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.6, 8.46], [-10.4, 8.64]]

    def _function(self,x):
        v = numpy.zeros(5)
        v[0] = 1. + x[0]
        v[0] += x[1]
        v[1] = v[0] * v[0]
        v[0] = x[0] * x[0]
        v[2] = 3. * v[0]
        v[2] += 19.
        v[0] = -14. * x[0]
        v[2] += v[0]
        v[0] = 6. * x[0]
        v[3] = v[0] * x[1]
        v[2] += v[3]
        v[3] = -14. * x[1]
        v[2] += v[3]
        v[3] = x[1] * x[1]
        v[0] = 3. * v[3]
        v[2] += v[0]
        v[0] = v[1] * v[2]
        v[1] = 1. + v[0]
        v[0] = 2. * x[0]
        v[2] = -3. * x[1]
        v[3] = v[0] + v[2]
        v[0] = v[3] * v[3]
        v[3] = x[0] * x[0]
        v[2] = 12. * v[3]
        v[2] += 18.;
        v[3] = -32. * x[0]
        v[2] += v[3]
        v[3] = 36. * x[0]
        v[4] = v[3] * x[1]
        v[3] = v[2] - v[4]
        v[2] = 48. * x[1]
        v[3] += v[2]
        v[2] = x[1] * x[1]
        v[4] = 27. * v[2]
        v[3] += v[4]
        v[4] = v[0] * v[3]
        v[0] = 30. + v[4]
        v[4] = v[1] * v[0]
        return v[4]

# new
class ex8_1_4(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = 12. * v[0]
        v[0] = pow(x[0], 4)
        v[2] = -6.3 * v[0]
        v[1] += v[2]
        v[2] = pow(x[0], 6)
        v[1] += v[2]
        v[2] = 6. * x[0]
        v[0] = v[2] * x[1]
        v[2] = -v[0]
        v[1] += v[2]
        v[2] = x[1] * x[1]
        v[0] = 6. * v[2]
        v[1] += v[0]
        return v[1]
# new
class ex8_1_5(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9101579868, 9.08085781188], [-10.7126564026, 8.35860923766]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = 4. * v[0]
        v[0] = pow(x[0], 4)
        v[2] = -2.1 * v[0]
        v[1] += v[2]
        v[2] = pow(x[0], 6)
        v[0] = 0.333333333333333 * v[2]
        v[1] += v[0]
        v[0] = x[0] * x[1]
        v[1] += v[0]
        v[0] = x[1] * x[1]
        v[2] = -4. * v[0]
        v[1] += v[2]
        v[2] = pow(x[1], 4)
        v[0] = 4. * v[2]
        v[1] += v[0]

        return v[1]
# new
class ex8_1_6(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.0000519964, 12.59995320324], [-6.0000519964, 12.59995320324]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -4. + x[0]
        v[1] = v[0] * v[0]
        v[1] += 0.1;
        v[0] = -4. + x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        if not (v[1] == 0):
            v[2] = -1. / v[1]
        else:
            v[2] = -1
        v[1] = -1. + x[0]
        v[0] = v[1] * v[1]
        v[0] += 0.2;
        v[1] = -1. + x[1]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        if not (v[0] == 0):
            v[3] = 1. / v[0]
        else:
            v[3] = 1
        v[0] = -v[3]
        v[2] += v[0]
        v[0] = -8. + x[0]
        v[3] = v[0] * v[0]
        v[3] += 0.2;
        v[0] = -8. + x[1]
        v[1] = v[0] * v[0]
        v[3] += v[1]
        if (v[3] == 0):
            v[1] = 1. / v[3]
        else:
            v[1] = 1
        v[3] = -v[1]
        v[2] += v[3]
        return v[2]
# new
class expfit(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.5210154534, 9.43108609194], [-9.3116998055, 9.61947017505]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 0.25 * x[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[0] * v[1]
        v[1] = -0.25 + v[0]
        v[0] = v[1] * v[1]
        v[1] = 0.5 * x[1]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -0.5 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.75 * x[1]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -0.75 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = numpy.exp(x[1]);
        v[2] = x[0] * v[1]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.25 * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -1.25 + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.5 * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -1.5 + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.75 * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -1.75 + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 2. * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -2. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 2.25 * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -2.25 + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 2.5 * x[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[0] * v[1]
        v[1] = -2.5 + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        return v[0]

# new
class gold(BenchmarkFunction):
    @property
    def domain(self):
        return [[-2.0, 2.0], [-2.0, 2.0]]

    def _function(self,x):
        v = numpy.zeros(5)
        v[0] = 1. + x[0]
        v[0] += x[1]
        v[1] = v[0] * v[0]
        v[0] = -14. * x[0]
        v[0] += 19.;
        v[2] = x[0] * x[0]
        v[3] = 3. * v[2]
        v[0] += v[3]
        v[3] = -14. * x[1]
        v[0] += v[3]
        v[3] = 6. * x[0]
        v[2] = v[3] * x[1]
        v[0] += v[2]
        v[2] = x[1] * x[1]
        v[3] = 3. * v[2]
        v[0] += v[3]
        v[3] = v[1] * v[0]
        v[1] = 1. + v[3]
        v[3] = 2. * x[0]
        v[0] = -3. * x[1]
        v[2] = v[3] + v[0]
        v[3] = v[2] * v[2]
        v[2] = -32. * x[0]
        v[2] += 18.;
        v[0] = x[0] * x[0]
        v[4] = 12. * v[0]
        v[2] += v[4]
        v[4] = 48. * x[1]
        v[2] += v[4]
        v[4] = 36. * x[0]
        v[0] = v[4] * x[1]
        v[4] = v[2] - v[0]
        v[2] = x[1] * x[1]
        v[0] = 27. * v[2]
        v[2] = v[4] + v[0]
        v[4] = v[3] * v[2]
        v[3] = 30. + v[4]
        v[4] = v[1] * v[3]
        return v[4]
# new
class griewank(BenchmarkFunction):
    @property
    def domain(self):
        return [[-100.0, 90.0], [-100.0, 90.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = 0.005 * v[0]
        v[0] = x[1] * x[1]
        v[2] = 0.005 * v[0]
        v[1] += v[2]
        v[2] = numpy.cos(x[0]);
        v[0] = x[1] / 1.4142135623730951;
        v[3] = numpy.cos(v[0]);
        v[0] = v[2] * v[3]
        v[2] = -v[0]
        v[1] += v[2]
        v[1] += 1.;
        return v[1]
# new
class hairy(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 7. * x[0]
        v[1] = numpy.sin(v[0]);
        v[0] = v[1] * v[1]
        v[1] = 7. * x[1]
        v[2] = numpy.cos(v[1]);
        v[1] = v[2] * v[2]
        v[2] = v[0] * v[1]
        v[0] = 30. * v[2]
        v[2] = x[0] - x[1]
        v[1] = v[2] * v[2]
        v[2] = 0.01 + v[1]
        v[1] = numpy.sqrt(v[2]);
        v[2] = 100. * v[1]
        v[0] += v[2]
        v[2] = x[0] * x[0]
        v[1] = 0.01 + v[2]
        v[2] = numpy.sqrt(v[1]);
        v[1] = 100. * v[2]
        v[0] += v[1]
        return v[0]
# new
class himmelbb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.7002462348, 9.26977838868], [-9.29902796, 9.630874836]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[1]
        v[1] = 1. - x[0]
        v[2] = v[0] * v[1]
        v[0] = 1. - x[1]
        v[1] = pow(x[0], 5)
        v[3] = 1. - v[1]
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = v[2] * v[3]
        v[2] = v[0] * v[0]
        return v[2]
# new
class himmelbg(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -x[0]
        v[1] = v[0] - x[1]
        v[0] = numpy.exp(v[1]);
        v[1] = x[0] * x[0]
        v[2] = 2. * v[1]
        v[1] = x[1] * x[1]
        v[3] = 3. * v[1]
        v[1] = v[2] + v[3]
        v[2] = v[0] * v[1]

        return v[2]

class himmelp1(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 95], [0, 75]]
        
    def _function(self,x):
        return (3.8112755343*x[0] -
                (0.1269366345*x[0]**2 - 0.0020567665*x[0]**3 +
                 1.0345e-5*x[0]**4 +
                 (0.0302344793*x[0] - 0.0012813448*x[0]**2 +
                  3.52599e-5*x[0]**3 - 2.266e-7*x[0]**4)*x[1] +
                 0.2564581253*x[1]**2 - 0.003460403*x[1]**3 +
                 1.35139e-5*x[1]**4 - 28.1064434908/(1 +x[1]) +
                 (3.405462e-4*x[0] - 5.2375e-6*x[0]**2-
                  6.3e-9*x[0]**3)*(x[1]**2) +
                 (7e-10*x[0]**3 - 1.6638e-6*x[0])*x[1]**3 -
                 2.8673112392*numpy.exp(5e-4*x[0]*x[1]))+
                6.8306567613*x[1] -75.1963666677)

# new
class hs001(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000086, 10.9999999914], [-1.5, 10.9999999828]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]

class hs002(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.7756292513, 11.2243707487], [1.5, 11.5]]

    def _function(self,x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

class hs003(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]

    def _function(self,x):
        return 1e-5*(x[1]-x[0])**2+x[1]

class hs004(BenchmarkFunction):
    @property
    def domain(self):
        return [[1, 11], [0, 10]]

    def _function(self,x):
        return 0.333333333333333*(1+x[0])**3+x[1]

# new
class hs005(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.5, 4.0], [-3.0, 3.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] + x[1]
        v[1] = numpy.sin(v[0]);
        v[0] = x[0] - x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[1] += 1.;
        rv = v[1] + -1.5*x[0]
        rv += 2.5*x[1]
        return rv;

class hs3mod(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 10], [0, 10]]

    def _function(self,x):
        return (x[1]-x[0])**2 + x[1]

# new
class hs5(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.5, 4.0], [-3.0, 3.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] + x[1]
        v[1] = numpy.sin(v[0]);
        v[0] = x[0] - x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[1] += 1.;
        rv = v[1] + -1.5*x[0]
        rv += 2.5*x[1]
        return rv;
# new
class humps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0000000029, 8.99999999739], [-10.000000004, 8.9999999964]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = 0.05 * v[0]
        v[0] = x[1] * x[1]
        v[2] = 0.05 * v[0]
        v[1] += v[2]
        v[2] = 20. * x[0]
        v[0] = numpy.sin(v[2]);
        v[2] = 20. * x[1]
        v[3] = numpy.sin(v[2]);
        v[2] = v[0] * v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        return v[1]

class jensmp(BenchmarkFunction):
    @property
    def domain(self):
        return [[0.1, 0.9], [0.1, 0.9]]

    def _function(self,x):
        return numpy.sum([((i+2)*2 - numpy.exp((i+1)*x[0]) -
                           numpy.exp((i+1)*x[1]))**2 for i in range(10)])

# new
class levy3(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 10.0], [-10.0, 10.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 2. * x[0]
        v[1] = 1. + v[0]
        v[0] = numpy.cos(v[1]);
        v[1] = 3. * x[0]
        v[2] = 2. + v[1]
        v[1] = numpy.cos(v[2]);
        v[2] = 2. * v[1]
        v[0] += v[2]
        v[2] = 4. * x[0]
        v[1] = 3. + v[2]
        v[2] = numpy.cos(v[1]);
        v[1] = 3. * v[2]
        v[0] += v[1]
        v[1] = 5. * x[0]
        v[2] = 4. + v[1]
        v[1] = numpy.cos(v[2]);
        v[2] = 4. * v[1]
        v[0] += v[2]
        v[2] = 6. * x[0]
        v[1] = 5. + v[2]
        v[2] = numpy.cos(v[1]);
        v[1] = 5. * v[2]
        v[0] += v[1]
        v[1] = 2. * x[1]
        v[2] = 1. + v[1]
        v[1] = numpy.cos(v[2]);
        v[2] = 3. * x[1]
        v[3] = 2. + v[2]
        v[2] = numpy.cos(v[3]);
        v[3] = 2. * v[2]
        v[1] += v[3]
        v[3] = 4. * x[1]
        v[2] = 3. + v[3]
        v[3] = numpy.cos(v[2]);
        v[2] = 3. * v[3]
        v[1] += v[2]
        v[2] = 5. * x[1]
        v[3] = 4. + v[2]
        v[2] = numpy.cos(v[3]);
        v[3] = 4. * v[2]
        v[1] += v[3]
        v[3] = 6. * x[1]
        v[2] = 5. + v[3]
        v[3] = numpy.cos(v[2]);
        v[2] = 5. * v[3]
        v[1] += v[2]
        v[2] = v[0] * v[1]
        v[0] = -v[2]
        return v[0]
# new
class loghairy(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9999999999, 9.00000000009], [-9.9999999974, 9.00000000234]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 7. * x[0]
        v[1] = numpy.sin(v[0]);
        v[0] = v[1] * v[1]
        v[1] = 7. * x[1]
        v[2] = numpy.cos(v[1]);
        v[1] = v[2] * v[2]
        v[2] = v[0] * v[1]
        v[0] = 30. * v[2]
        v[0] += 100.;
        v[2] = x[0] - x[1]
        v[1] = v[2] * v[2]
        v[2] = 0.01 + v[1]
        v[1] = numpy.sqrt(v[2]);
        v[2] = 100. * v[1]
        v[0] += v[2]
        v[2] = x[0] * x[0]
        v[1] = 0.01 + v[2]
        v[2] = numpy.sqrt(v[1]);
        v[1] = 100. * v[2]
        v[0] += v[1]
        v[1] = v[0] / 100.;
        v[0] = numpy.log(v[1]);
        return v[0]


class logros(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 11], [0, 11]]

    def _function(self,x):
        return numpy.log(1+10000*(x[1]-x[0]**2)**2+(1-x[0])**2)

# new
class maratosb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.000000125, 8.0999998875], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] * x[1]
        v[2] = v[0] + v[1]
        v[0] = -1. + v[2]
        v[2] = v[0] * v[0]
        v[0] = 1.e+06 * v[2]
        rv = v[0] + x[0]
        return rv;

class mdhole(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [-10, 10]]
        
    def _function(self,x):
        return 100*(numpy.sin(x[0])-x[1])**2+x[0]

class median_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[1e-08, 10.00000001], [-9.499789331, 10.500210669]]

    def _function(self,x):
        term1 = [-0.171747132,-0.843266708,-0.550375356,-0.301137904,
                 -0.292212117,-0.224052867,-0.349830504,-0.856270347,
                 -0.067113723,-0.500210669,-0.998117627,-0.578733378,
                 -0.991133039,-0.762250467,-0.130692483,-0.639718759,
                 -0.159517864,-0.250080533,-0.668928609]
        return x[0] + numpy.sum(numpy.sqrt(x[0]**2 + (term1 + x[1])**2))

# new
class mexhat(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.1417744688, 9.77240297808], [-9.2634512042, 9.66289391622]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -1. + x[0]
        v[1] = v[0] * v[0]
        v[0] = -2. * v[1]
        v[1] = x[0] * x[0]
        v[2] = x[1] - v[1]
        v[1] = v[2] * v[2]
        v[2] = v[1] / 10000.;
        v[2] += -0.02;
        v[1] = -1. + x[0]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = 10000. * v[3]
        v[3] = v[0] + v[2]

        return v[3]
# new
class nasty(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 5.e+09 * x[0]
        v[1] = 1.e+10 * x[0]
        v[2] = v[0] * v[1]
        v[0] = 0.5 * x[1]
        v[1] = v[0] * x[1]
        v[0] = v[2] + v[1]
        return v[0]
# new
class price(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = pow(x[0], 3)
        v[1] = 2. * v[0]
        v[0] = v[1] * x[1]
        v[1] = pow(x[1], 3)
        v[2] = v[0] - v[1]
        v[0] = v[2] * v[2]
        v[2] = 6. * x[0]
        v[1] = x[1] * x[1]
        v[3] = v[2] - v[1]
        v[2] = v[3] + x[1]
        v[3] = v[2] * v[2]
        v[2] = v[0] + v[3]

        return v[2]
# new
class rbrock(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 5.0], [-10.0, 10.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class rosenbr(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 5.0], [-10.0, 10.0]]

    def _function(self,x):
        v = numpy.zeros(7)
        v[0] = x[0] * x[0]
        v[5] = -10. * v[0]
        v[5] = v[5] + 10.*x[1]
        v[6] = 1. - x[0]
        v[2] = v[5] * v[5]
        v[3] = v[6] * v[6]
        v[4] = v[2] + v[3]
        return v[4]
# new
class s201(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 13.5], [-4.0, 14.4]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -5. + x[0]
        v[1] = v[0] * v[0]
        v[0] = 4. * v[1]
        v[1] = -6. + x[1]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]
# new
class s202(BenchmarkFunction):
    @property
    def domain(self):
        return [[-4.9999999725, 13.50000002475], [-6.0000000005, 12.59999999955]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -13. + x[0]
        v[1] = -2. * x[1]
        v[0] += v[1]
        v[1] = x[1] * x[1]
        v[2] = 5. * v[1]
        v[0] += v[2]
        v[2] = pow(x[1], 3)
        v[1] = v[0] - v[2]
        v[0] = v[1] * v[1]
        v[1] = -29. + x[0]
        v[2] = -14. * x[1]
        v[1] += v[2]
        v[2] = x[1] * x[1]
        v[1] += v[2]
        v[2] = pow(x[1], 3)
        v[1] += v[2]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]
# new
class s204(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10000.0, 10000.0], [-10000.0, 10000.0]]

    def _function(self,x):
        v = numpy.zeros(11)
        v[0] = 5.66598 * x[0]
        v[1] = 2.77141 * x[1]
        v[2] = v[0] + v[1]
        v[0] = x[0] * v[2]
        v[2] = 1.2537 * v[0]
        v[0] = 2.77141 * x[0]
        v[1] = 2.12413 * x[1]
        v[3] = v[0] + v[1]
        v[0] = x[1] * v[3]
        v[3] = 1.2537 * v[0]
        v[8] = v[2] + v[3]
        v[8] += 0.13294;
        v[8] = v[8] - 0.564255*x[0]
        v[8] += 0.392417*x[1]
        
        v[3] = 5.66598 * x[0]
        v[0] = 2.77141 * x[1]
        v[1] = v[3] + v[0]
        v[3] = x[0] * v[1]
        v[1] = -0.682005 * v[3]
        v[3] = 2.77141 * x[0]
        v[0] = 2.12413 * x[1]
        v[4] = v[3] + v[0]
        v[3] = x[1] * v[4]
        v[4] = -0.682005 * v[3]
        v[9] = v[1] + v[4]
        v[9] += -0.244378;
        v[9] = v[9] - 0.404979*x[0]
        v[9] += 0.927589*x[1]
        
        v[4] = 5.66598 * x[0]
        v[3] = 2.77141 * x[1]
        v[0] = v[4] + v[3]
        v[4] = x[0] * v[0]
        v[0] = 0.51141 * v[4]
        v[4] = 2.77141 * x[0]
        v[3] = 2.12413 * x[1]
        v[5] = v[4] + v[3]
        v[4] = x[1] * v[5]
        v[5] = 0.51141 * v[4]
        v[10] = v[0] + v[5]
        v[10] += 0.325895;
        v[10] = v[10] - 0.0735084*x[0]
        v[10] += 0.535493*x[1]

        v[6] = v[8] * v[8]
        v[7] = v[9] * v[9]
        v[6] += v[7]
        v[7] = v[10] * v[10]
        v[6] += v[7]
        return v[6]
# new
class s205(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.0000000003, 11.69999999973], [-9.5000000001, 9.44999999991]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 1. - x[1]
        v[1] = x[0] * v[0]
        v[0] = 1.5 - v[1]
        v[1] = v[0] * v[0]
        v[0] = x[1] * x[1]
        v[2] = 1. - v[0]
        v[0] = x[0] * v[2]
        v[2] = 2.25 - v[0]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = pow(x[1], 3)
        v[2] = 1. - v[0]
        v[0] = x[0] * v[2]
        v[2] = 2.625 - v[0]
        v[0] = v[2] * v[2]
        v[1] += v[0]

        return v[1]
# new
class s206(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 1. - x[0]
        v[2] = v[1] * v[1]
        v[1] = 100. * v[2]
        v[2] = v[0] + v[1]
        return v[2]
# new
class s207(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000009, 9.89999999919], [-9.0000000021, 9.89999999811]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 1. - x[0]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]
# new
class s208(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0000000001, 9.89999999991]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s209(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 10000. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s210(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000407, 9.89999996337], [-9.0000000813, 9.89999992683]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 1.e+06 * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s211(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = pow(x[0], 3)
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s212(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(6)
        v[0] = x[0] + x[1]
        v[1] = 4. * v[0]
        v[0] = v[1] * v[1]
        v[1] = x[0] + x[1]
        v[2] = 4. * v[1]
        v[1] = x[0] - x[1]
        v[3] = -2. + x[0]
        v[4] = v[3] * v[3]
        v[3] = x[1] * x[1]
        v[5] = v[4] + v[3]
        v[4] = -1. + v[5]
        v[5] = v[1] * v[4]
        v[1] = v[2] + v[5]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]
# new
class s213(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.9315761266, 9.96158148606], [-8.9315761266, 9.96158148606]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] - x[1]
        v[1] = v[0] * v[0]
        v[0] = 10. * v[1]
        v[1] = -1. + x[0]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        v[0] = pow(v[1], 4)
        return v[0]
# new
class s214(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] - x[1]
        v[1] = v[0] * v[0]
        v[0] = 10. * v[1]
        v[1] = -1. + x[0]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        v[0] = pow(v[1], 0.25);
        return v[0]
# new
class s229(BenchmarkFunction):
    @property
    def domain(self):
        return [[-2.0, 2.0], [-2.0, 2.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s274(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0000000052, 8.99999999532],[-10.0000000026, 8.99999999766]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 0.5 * x[1]
        v[1] = x[0] + v[0]
        v[0] = x[0] * v[1]
        v[1] = 0.5 * x[0]
        v[2] = 0.3333333333333333 * x[1]
        v[3] = v[1] + v[2]
        v[1] = x[1] * v[3]
        v[3] = v[0] + v[1]
        return v[3]
# new
class s290(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = 2. * x[1]
        v[2] = x[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]
# new
class s308(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.8445627595, 9.13989351645], [-10.6945637774, 8.37489260034]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] * x[1]
        v[0] += v[1]
        v[1] = x[0] * x[1]
        v[0] += v[1]
        v[1] = v[0] * v[0]
        v[0] = numpy.sin(x[0]);
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = numpy.cos(x[1]);
        v[0] = v[2] * v[2]
        v[1] += v[0]

        return v[1]
# new
class s309(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.517315694, 12.1344158754], [-6.1, 12.51]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = pow(x[0], 4)
        v[1] = 1.41 * v[0]
        v[0] = pow(x[0], 3)
        v[2] = -12.76 * v[0]
        v[1] += v[2]
        v[2] = x[0] * x[0]
        v[0] = 39.91 * v[2]
        v[1] += v[0]
        v[0] = -3.9 + x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[1] += 24.37;
        rv = v[1] + -51.93*x[0]
        return rv;
# new
class s311(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.0, 11.7], [-8.0, 10.8]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = v[0] + x[1]
        v[0] = -11. + v[1]
        v[1] = v[0] * v[0]
        v[0] = x[1] * x[1]
        v[2] = x[0] + v[0]
        v[0] = -7. + v[2]
        v[2] = v[0] * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class s312(BenchmarkFunction):
    @property
    def domain(self):
        return [[-31.0266522627, -9.92398703643], [-46.7600087813, -24.08400790317]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = 12. * x[1]
        v[2] = v[0] + v[1]
        v[0] = -1. + v[2]
        v[2] = v[0] * v[0]
        v[0] = x[0] * x[0]
        v[1] = 49. * v[0]
        v[0] = x[1] * x[1]
        v[3] = 49. * v[0]
        v[1] += v[3]
        v[3] = 84. * x[0]
        v[1] += v[3]
        v[3] = 2324. * x[1]
        v[1] += v[3]
        v[3] = -681. + v[1]
        v[1] = v[3] * v[3]
        v[3] = v[2] + v[1]
        return v[3]

class s328(BenchmarkFunction):
    @property
    def domain(self):
        return [[1, 2.7], [1, 2.7]]

    def _function(self,x):
        return (0.1*(x[0]**2 + (1 + (x[1]**2))/(x[0]**2) +
                    (100 + (x[0]**2)*(x[1]**2))/(x[0]**4*x[1]**4)) + 1.2)

# new
class s386(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 13.5], [-4.0, 14.4]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -5. + x[0]
        v[1] = v[0] * v[0]
        v[0] = 4. * v[1]
        v[1] = -6. + x[1]
        v[2] = v[1] * v[1]
        v[1] = v[0] + v[2]
        return v[1]

class sim2bqp(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 9], [0, 0.45]]

    def _function(self,x):
        return (x[1]-x[0])**2+x[1]+(x[0]+x[1])**2

class simbqp(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10, 9], [0, 0.45]]

    def _function(self,x):
        return (x[1]-x[0])**2+x[1]+(2*x[0]+x[1])**2

# new
class sineval(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0000000002, 8.99999999982], [-10.0000000002, 8.99999999982]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = numpy.sin(x[0]);
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 1000. * v[0]
        v[0] = x[0] * x[0]
        v[2] = 0.25 * v[0]
        v[0] = v[1] + v[2]
        return v[0]
# new
class sisser(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9978640372, 9.00192236652], [-9.9983980285, 9.00144177435]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = pow(x[0], 4)
        v[1] = 3. * v[0]
        v[0] = x[0] * x[1]
        v[2] = v[0] * v[0]
        v[0] = -2. * v[2]
        v[1] += v[0]
        v[0] = pow(x[1], 4)
        v[2] = 3. * v[0]
        v[1] += v[2]
        return v[1]
# new
class st_e39(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.0000519964, 12.59995320324], [-6.0000519964, 12.59995320324]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -4. + x[0]
        v[1] = v[0] * v[0]
        v[1] += 0.1;
        v[0] = -4. + x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        if not (v[1] == 0.):
            v[2] = -1. / v[1]
        else:
            v[2] = -1
        v[1] = -1. + x[0]
        v[0] = v[1] * v[1]
        v[0] += 0.2;
        v[1] = -1. + x[1]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        if not (v[0] == 0.):
            v[3] = 1. / v[0]
        else:
            v[3] = 1
        v[0] = -v[3]
        v[2] += v[0]
        v[0] = -8. + x[0]
        v[3] = v[0] * v[0]
        v[3] += 0.2;
        v[0] = -8. + x[1]
        v[1] = v[0] * v[0]
        v[3] += v[1]
        if not (v[3] == 0.):
            v[1] = 1. / v[3]
        else:
            v[1] = 1
        v[3] = -v[1]
        v[2] += v[3]
        return v[2]
# new
class tre(BenchmarkFunction):
    @property
    def domain(self):
        return [[-5.0, 5.0], [-5.0, 5.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = pow(x[0], 4)
        v[1] = pow(x[0], 3)
        v[2] = 4. * v[1]
        v[0] += v[2]
        v[2] = x[0] * x[0]
        v[1] = 4. * v[2]
        v[0] += v[1]
        v[1] = x[1] * x[1]
        v[0] += v[1]
        return v[0]
# new
class zangwil2(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.0, 12.6], [-1.0, 17.1]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = 1.0666666666666667 * v[0]
        v[0] = x[1] * x[1]
        v[2] = 1.0666666666666667 * v[0]
        v[1] += v[2]
        v[2] = 8. * x[0]
        v[0] = v[2] * x[1]
        v[2] = -0.06666666666666667 * v[0]
        v[1] += v[2]
        v[1] += 66.06666666666666;
        rv = v[1] + -3.7333333333333334*x[0]
        rv += -17.066666666666666*x[1]
        return rv;

class allinit(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.1426691153, 8.8573308847], [1, 11.2456257795],[-1e10, 1]]
        
    def _function(self,x):
        return (x[0]**2 + x[1]**2 + (x[2] + 2)**2 + x[2] + numpy.sin(x[2])**2 +
                x[0]**2*x[1]**2 + numpy.sin(x[2])**2 + x[1]**4 +
                (-4 + numpy.sin(2)**2 + x[1]**2*x[2]**2 + x[0])**2 +
                (x[2]**2 + (x[0] + 2)**2)**2 + numpy.sin(2)**4 - 1)

# new
class biggs3(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000168, 9.89999998488], [-1.96500000271271e-07, 17.99999982315], [-5.0000001349, 13.49999987859]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -0.1 * x[0]
        v[1] = numpy.exp(v[0]);
        v[0] = -1.0764003502856656 + v[1]
        v[1] = -0.1 * x[1]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[2] = v[0] - v[1]
        v[0] = 2.0109601381069178 + v[2]
        v[2] = v[0] * v[0]
        v[0] = -0.2 * x[0]
        v[1] = numpy.exp(v[0]);
        v[0] = -1.490041229246583 + v[1]
        v[1] = -0.2 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[2] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 1.3479868923516647 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = -0.30000000000000004 * x[0]
        v[0] = numpy.exp(v[3]);
        v[3] = -1.3954655145790045 + v[0]
        v[0] = -0.30000000000000004 * x[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[2] * v[1]
        v[1] = v[3] - v[0]
        v[3] = 0.9035826357366061 + v[1]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.4 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -1.1844314055759346 + v[3]
        v[3] = -0.4 * x[1]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[1] - v[3]
        v[1] = 0.6056895539839662 + v[0]
        v[0] = v[1] * v[1]
        v[2] += v[0]
        v[0] = -0.5 * x[0]
        v[1] = numpy.exp(v[0]);
        v[0] = -0.9788467744270443 + v[1]
        v[1] = -0.5 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[2] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.4060058497098381 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = -0.6000000000000001 * x[0]
        v[0] = numpy.exp(v[3]);
        v[3] = -0.8085717350789321 + v[0]
        v[0] = -0.6000000000000001 * x[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[2] * v[1]
        v[1] = v[3] - v[0]
        v[3] = 0.2721538598682374 + v[1]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.7000000000000001 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.6744560818392907 + v[3]
        v[3] = -0.7000000000000001 * x[1]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[1] - v[3]
        v[1] = 0.18243018787565385 + v[0]
        v[0] = v[1] * v[1]
        v[2] += v[0]
        v[0] = -0.8 * x[0]
        v[1] = numpy.exp(v[0]);
        v[0] = -0.5699382629128076 + v[1]
        v[1] = -0.8 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[2] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.12228661193509861 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = -0.9 * x[0]
        v[0] = numpy.exp(v[3]);
        v[3] = -0.4879237780620434 + v[0]
        v[0] = -0.9 * x[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[2] * v[1]
        v[1] = v[3] - v[0]
        v[3] = 0.0819711673418777 + v[1]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.4225993581888325 + v[3]
        v[3] = -x[1]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[1] - v[3]
        v[1] = 0.05494691666620255 + v[0]
        v[0] = v[1] * v[1]
        v[2] += v[0]
        v[0] = -1.1 * x[0]
        v[1] = numpy.exp(v[0]);
        v[0] = -0.3696195949033336 + v[1]
        v[1] = -1.1 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[2] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.03683201970920533 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = -1.2000000000000002 * x[0]
        v[0] = numpy.exp(v[3]);
        v[3] = -0.3258527319974954 + v[0]
        v[0] = -1.2000000000000002 * x[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[2] * v[1]
        v[1] = v[3] - v[0]
        v[3] = 0.024689241147060066 + v[1]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -1.3 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.28907018464926004 + v[3]
        v[3] = -1.3 * x[1]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[1] - v[3]
        v[1] = 0.01654969326228231 + v[0]
        v[0] = v[1] * v[1]
        v[2] += v[0]
        return v[2]

class _box(BenchmarkFunction):
    def _function(self,x):
        coeffs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1]
        coeff2 = [0.536957976864517,0.683395469841369,0.691031152313854,
                  0.652004407146905,0.599792712713548,0.54633288391736,
                  0.495673421825855,0.448993501489319,0.406446249936512,
                  0.36783404124168]
        return numpy.sum((numpy.exp(numpy.multiply(coeffs,x[0]))-
                          numpy.exp(numpy.multiply(coeffs,x[1]))-
                          numpy.multiply(coeff2,x[2]))**2)
   
class box3(_box):
    @property
    def domain(self):
        return [[-9.0000004305, 9.89999961255],
                [3.23989999984065e-06, 18.00000291591],
                [-8.9999997323, 9.90000024093]]

# new
class denschnd(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0002179404, 8.99980385364], [-9.9733994128, 9.02394052848], [-10.0001458391, 8.99986874481]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = pow(x[1], 3)
        v[2] = v[0] + v[1]
        v[0] = pow(x[2], 4)
        v[1] = v[2] - v[0]
        v[2] = v[1] * v[1]
        v[1] = 2. * x[0]
        v[0] = v[1] * x[1]
        v[1] = v[0] * x[2]
        v[0] = v[1] * v[1]
        v[2] += v[0]
        v[0] = 2. * x[0]
        v[1] = v[0] * x[1]
        v[0] = 3. * x[1]
        v[3] = v[0] * x[2]
        v[0] = v[1] - v[3]
        v[1] = x[0] * x[2]
        v[3] = v[0] + v[1]
        v[0] = v[3] * v[3]
        v[2] += v[0]

        return v[2]
# new
class denschne(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] * x[1]
        v[2] = x[1] + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = numpy.exp(x[2]);
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        return v[0]

class eg1(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.2302657121, 9.7697342879], [-1, 1], [1, 2]]
        
    def _function(self,x):
        return (x[0]**2 + (x[1]*x[2])**4 + x[0]*x[2] +
                numpy.sin(x[0]+x[2])*x[1] + x[1])

# new
class engval2(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = x[1] * x[1]
        v[0] += v[1]
        v[1] = x[2] * x[2]
        v[0] += v[1]
        v[1] = -1. + v[0]
        v[0] = v[1] * v[1]
        v[1] = x[0] * x[0]
        v[2] = x[1] * x[1]
        v[1] += v[2]
        v[2] = -2. + x[2]
        v[3] = v[2] * v[2]
        v[1] += v[3]
        v[3] = -1. + v[1]
        v[1] = v[3] * v[3]
        v[0] += v[1]
        v[1] = x[0] + x[1]
        v[1] += x[2]
        v[3] = -1. + v[1]
        v[1] = v[3] * v[3]
        v[0] += v[1]
        v[1] = x[0] + x[1]
        v[3] = v[1] - x[2]
        v[1] = 1. + v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = x[1] * x[1]
        v[1] = 3. * v[3]
        v[3] = pow(x[0], 3)
        v[1] += v[3]
        v[3] = 5. * x[2]
        v[2] = v[3] - x[0]
        v[3] = 1. + v[2]
        v[2] = v[3] * v[3]
        v[1] += v[2]
        v[2] = -36. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        return v[0]

class fermat_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.9999999999, 12.0000000001],
                [-8.8452994616, 11.1547005384], [1e-08, 10.00000001]]

    def _function(self,x):
        return (numpy.sqrt(x[2]**2 + x[0]**2 + x[1]**2) +
                numpy.sqrt(x[2]**2 + (x[0] - 4)**2 + x[1]**2)+
                numpy.sqrt(x[2]**2 + (x[0] - 2)**2 + (x[1] - 4)**2) + x[2])

class fermat2_vareps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8, 12], [-9.00000002, 10.99999998], [1e-08, 10.00000001]]

    def _function(self,x):
        return (numpy.sqrt(x[2]**2 + x[0]**2 + x[1]**2) +
                numpy.sqrt(x[2]**2 + (x[0] - 4)**2 + x[1]**2)+
                numpy.sqrt(x[2]**2 + (x[0] - 2)**2 + (x[1] - 1)**2) + x[2])

# new
class growth(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.53967303, 10.314294273], [-9.5571937886, 9.39852559026], [-9.836246351, 9.1473782841]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 2.0794415416798357 * x[2]
        v[1] = x[1] + v[0]
        v[0] = 2.079442 * v[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[0] * v[1]
        v[1] = -8. + v[0]
        v[0] = v[1] * v[1]
        v[1] = 2.1972245773362196 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.197225 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -8.4305 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.302585092994046 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.302585 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -9.5294 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.3978952727983707 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.397895 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -10.4627 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.4849066497880004 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.484907 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -12. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.5649493574615367 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.564949 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -13.0205 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.6390573296152584 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.639057 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -14.5949 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.70805020110221 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.70805 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -16.1078 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.772588722239781 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.772589 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -18.0596 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.8903717578961645 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.890372 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -20.4569 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 2.995732273553991 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.995732 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -24.25 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 3.2188758248682006 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 3.218876 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -32.9863 + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        return v[0]
# new
class growthls(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.53967303, 10.314294273], [-9.5571937886, 9.39852559026], [-9.836246351, 9.1473782841]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 2.079441542 * x[2]
        v[1] = x[1] + v[0]
        v[0] = 2.079442 * v[1]
        v[1] = numpy.exp(v[0]);
        v[0] = x[0] * v[1]
        v[1] = -8. + v[0]
        v[0] = 2.079441542 * x[2]
        v[2] = x[1] + v[0]
        v[0] = 2.079442 * v[2]
        v[2] = numpy.exp(v[0]);
        v[0] = x[0] * v[2]
        v[2] = -8. + v[0]
        v[0] = v[1] * v[2]
        v[1] = 2.19722457733622 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.19722457733622 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -8.4305 + v[1]
        v[1] = 2.19722457733622 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.19722457733622 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -8.4305 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.302585093 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.302585 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -9.5294 + v[1]
        v[1] = 2.302585093 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.302585 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -9.5294 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.397895273 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.397895 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -10.4627 + v[1]
        v[1] = 2.397895273 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.397895 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -10.4627 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.48490665 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.484907 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -12. + v[1]
        v[1] = 2.48490665 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.484907 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -12. + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.564949357 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.564949 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -13.0205 + v[1]
        v[1] = 2.564949357 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.564949 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -13.0205 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.63905733 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.639057 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -14.5949 + v[1]
        v[1] = 2.63905733 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.639057 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -14.5949 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.708050201 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.70805 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -16.1078 + v[1]
        v[1] = 2.708050201 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.70805 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -16.1078 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.77258872223978 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.772589 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -18.0596 + v[1]
        v[1] = 2.77258872223978 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.772589 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -18.0596 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.890371758 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.890372 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -20.4569 + v[1]
        v[1] = 2.890371758 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.890372 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -20.4569 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 2.99573227355399 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 2.995732 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -24.25 + v[1]
        v[1] = 2.99573227355399 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 2.995732 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -24.25 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        v[1] = 3.218875825 * x[2]
        v[2] = x[1] + v[1]
        v[1] = 3.218876 * v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[0] * v[2]
        v[2] = -32.9863 + v[1]
        v[1] = 3.218875825 * x[2]
        v[3] = x[1] + v[1]
        v[1] = 3.218876 * v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = -32.9863 + v[1]
        v[1] = v[2] * v[3]
        v[0] += v[1]
        return v[0]
# new
class hatfldd(BenchmarkFunction):
    @property
    def domain(self):
        return [[-6.8005223172, 11.87952991452], [-11.0178316754, 8.08395149214], [-10.7584301644, 8.31741285204]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 0.2 * x[2]
        v[1] = numpy.exp(v[0]);
        v[0] = 0.2 * x[1]
        v[2] = numpy.exp(v[0]);
        v[0] = x[0] * v[2]
        v[2] = v[1] - v[0]
        v[1] = 1.751 + v[2]
        v[2] = v[1] * v[1]
        v[1] = 0.3 * x[2]
        v[0] = numpy.exp(v[1]);
        v[1] = 0.3 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 1.561 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 0.4 * x[2]
        v[0] = numpy.exp(v[3]);
        v[3] = 0.4 * x[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[0] * v[1]
        v[1] = v[0] - v[3]
        v[0] = 1.391 + v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.5 * x[2]
        v[0] = numpy.exp(v[1]);
        v[1] = 0.5 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 1.239 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 0.6 * x[2]
        v[0] = numpy.exp(v[3]);
        v[3] = 0.6 * x[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[0] * v[1]
        v[1] = v[0] - v[3]
        v[0] = 1.103 + v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.7 * x[2]
        v[0] = numpy.exp(v[1]);
        v[1] = 0.7 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.981 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 0.75 * x[2]
        v[0] = numpy.exp(v[3]);
        v[3] = 0.75 * x[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[0] * v[1]
        v[1] = v[0] - v[3]
        v[0] = 0.925 + v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.8 * x[2]
        v[0] = numpy.exp(v[1]);
        v[1] = 0.8 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.8721 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 0.85 * x[2]
        v[0] = numpy.exp(v[3]);
        v[3] = 0.85 * x[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[0] * v[1]
        v[1] = v[0] - v[3]
        v[0] = 0.8221 + v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.9 * x[2]
        v[0] = numpy.exp(v[1]);
        v[1] = 0.9 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = x[0] * v[3]
        v[3] = v[0] - v[1]
        v[0] = 0.7748 + v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        return v[2]

class least(BenchmarkFunction):
    @property
    def domain(self):
        return [[473.98605675534, 506.6511741726],
                [-159.3518936954, -125.41670432586],[-5, 4.5]]

    def _function(self,x):
        term0 = [127,151,379,421,460,426]
        term1 = [-5,-3,-1,5,3,1]
        return numpy.sum((term0 - x[1]*numpy.exp(numpy.multiply(term1,x[2])) -
                          x[0])**2)

# new
class s240(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 9.0], [-10.0, 9.0], [-10.0, 9.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] - x[1]
        v[1] = v[0] + x[2]
        v[0] = v[1] * v[1]
        v[1] = -x[0]
        v[1] += x[1]
        v[1] += x[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = x[0] + x[1]
        v[1] = v[2] - x[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        return v[0]

class s242(_box):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10]]

# new
class s243(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10000.0, 10000.0], [-10000.0, 10000.0], [-10000.0, 10000.0]]

    def _function(self,x):
        v = numpy.zeros(11)
        v[0] = 2.95137 * x[0]
        v[1] = 4.87407 * x[1]
        v[0] += v[1]
        v[1] = -2.0506 * x[2]
        v[0] += v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.87584 * v[1]
        v[1] = 2.95137 * x[0]
        v[2] = 4.87407 * x[1]
        v[1] += v[2]
        v[2] = -2.0506 * x[2]
        v[1] += v[2]
        v[2] = x[1] * v[1]
        v[1] = 0.87584 * v[2]
        v[7] = v[0] + v[1]
        v[1] = 2.95137 * x[0]
        v[2] = 4.87407 * x[1]
        v[1] += v[2]
        v[2] = -2.0506 * x[2]
        v[1] += v[2]
        v[2] = x[2] * v[1]
        v[1] = 0.87584 * v[2]
        v[7] += v[1]
        v[7] += 0.14272;
        v[7] = v[7] - 0.564255*x[0]
        v[7] += 0.392417*x[1]
        v[7] -= 0.404979*x[2]

        v[1] = 4.87407 * x[0]
        v[2] = 9.39321 * x[1]
        v[1] += v[2]
        v[2] = -3.93181 * x[2]
        v[1] += v[2]
        v[2] = x[0] * v[1]
        v[1] = -0.675975 * v[2]
        v[2] = 4.87407 * x[0]
        v[3] = 9.39321 * x[1]
        v[2] += v[3]
        v[3] = -3.93181 * x[2]
        v[2] += v[3]
        v[3] = x[1] * v[2]
        v[2] = -0.675975 * v[3]
        v[8] = v[1] + v[2]
        v[2] = 4.87407 * x[0]
        v[3] = 9.39321 * x[1]
        v[2] += v[3]
        v[3] = -3.93181 * x[2]
        v[2] += v[3]
        v[3] = x[2] * v[2]
        v[2] = -0.675975 * v[3]
        v[8] += v[2]
        v[8] += -0.184918;
        v[8] = v[8] + 0.927589*x[0]
        v[8] -= 0.0735084*x[1]
        v[8] += 0.535493*x[2]

        v[2] = -2.0506 * x[0]
        v[3] = -3.93189 * x[1]
        v[2] += v[3]
        v[3] = 2.64745 * x[2]
        v[2] += v[3]
        v[3] = x[0] * v[2]
        v[2] = -0.239524 * v[3]
        v[3] = -2.0506 * x[0]
        v[4] = -3.93189 * x[1]
        v[3] += v[4]
        v[4] = 2.64745 * x[2]
        v[3] += v[4]
        v[4] = x[1] * v[3]
        v[3] = -0.239524 * v[4]
        v[9] = v[2] + v[3]
        v[3] = -2.0506 * x[0]
        v[4] = -3.93189 * x[1]
        v[3] += v[4]
        v[4] = 2.64745 * x[2]
        v[3] += v[4]
        v[4] = x[2] * v[3]
        v[3] = -0.239524 * v[4]
        v[9] += v[3]
        v[9] += -0.521869;
        v[9] = v[9] + 0.658799*x[0]
        v[9] -= 0.636666*x[1]
        v[9] -= 0.681091*x[2]

        v[10] = -0.685306 - 0.869487*x[0]
        v[10] += 0.586387*x[1]
        v[10] += 0.289826*x[2]

        v[5] = v[7] * v[7]
        v[6] = v[8] * v[8]
        v[5] += v[6]
        v[6] = v[9] * v[9]
        v[5] += v[6]
        v[6] = v[10] * v[10]
        v[5] += v[6]
        return v[5]

class s244(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10]]

    def _function(self,x):
        return ((0.934559787821252 + numpy.exp(-0.1*x[0]) -
                 numpy.exp(-0.1*x[1])*x[2])**2 +
                (-0.142054336894918 + numpy.exp(-0.2*x[0]) -
                 numpy.exp(-0.2*x[1])*x[2])**2 +
                (-0.491882878842398 + numpy.exp(-0.3*x[0]) -
                 numpy.exp(-0.3*x[1])*x[2])**2)

# new
class s245(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000001, 0.0], [-7.90000065364893e-09, 0.0], [-8.9999999998, 0.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] / 10.;
        v[1] = -v[0]
        v[0] = numpy.exp(v[1]);
        v[1] = x[1] / 10.;
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = v[0] - v[1]
        v[0] = -0.5369579768645172 * x[2]
        v[1] = v[2] + v[0]
        v[2] = v[1] * v[1]
        v[1] = -0.2 * x[0]
        v[0] = numpy.exp(v[1]);
        v[1] = -0.2 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = v[0] - v[3]
        v[0] = -0.6833954698413691 * x[2]
        v[3] = v[1] + v[0]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.3 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.3 * x[1]
        v[0] = numpy.exp(v[1]);
        v[1] = v[3] - v[0]
        v[3] = -0.6910311523138539 * x[2]
        v[0] = v[1] + v[3]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = -0.4 * x[0]
        v[0] = numpy.exp(v[1]);
        v[1] = -0.4 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = v[0] - v[3]
        v[0] = -0.6520044071469051 * x[2]
        v[3] = v[1] + v[0]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.5 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.5 * x[1]
        v[0] = numpy.exp(v[1]);
        v[1] = v[3] - v[0]
        v[3] = -0.599792712713548 * x[2]
        v[0] = v[1] + v[3]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = -0.6 * x[0]
        v[0] = numpy.exp(v[1]);
        v[1] = -0.6 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = v[0] - v[3]
        v[0] = -0.5463328839173601 * x[2]
        v[3] = v[1] + v[0]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.7 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.7 * x[1]
        v[0] = numpy.exp(v[1]);
        v[1] = v[3] - v[0]
        v[3] = -0.49567342182585505 * x[2]
        v[0] = v[1] + v[3]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = -0.8 * x[0]
        v[0] = numpy.exp(v[1]);
        v[1] = -0.8 * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = v[0] - v[3]
        v[0] = -0.44899350148931905 * x[2]
        v[3] = v[1] + v[0]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        v[1] = -0.9 * x[0]
        v[3] = numpy.exp(v[1]);
        v[1] = -0.9 * x[1]
        v[0] = numpy.exp(v[1]);
        v[1] = v[3] - v[0]
        v[3] = -0.4064462499365124 * x[2]
        v[0] = v[1] + v[3]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = -1. * x[0]
        v[0] = numpy.exp(v[1]);
        v[1] = -1. * x[1]
        v[3] = numpy.exp(v[1]);
        v[1] = v[0] - v[3]
        v[0] = -0.36783404124167984 * x[2]
        v[3] = v[1] + v[0]
        v[1] = v[3] * v[3]
        v[2] += v[1]
        return v[2]
# new
class s246(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0000000002, 9.89999999982], [-9.0000000002, 9.89999999982], [-9.0000000004, 9.89999999964]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] + x[1]
        v[1] = v[0] / 2.;
        v[0] = v[1] * v[1]
        v[1] = x[2] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 1. - x[1]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        return v[1]

# REALLY needs more precise floats
class s333(BenchmarkFunction):
    @property
    def domain(self):
        return [[79.901992908, 89.9117936172], [-1, 0.9], [-1, 0.9]]

    def _function(self,x):
        coeff1 = [-4,-5.75,-7.5,-24,-32,-48,-72,-96]
        coeff2 = [0.013869625520111,0.0152439024390244,0.0178890876565295,
                  0.0584795321637427,0.102040816326531,0.222222222222222,
                  0.769230769230769,1.66666666666667]
        y = 0
        for i in range(8):
            y += (1-coeff2[i]*numpy.exp(coeff1[i]*x[1])*x[0]-coeff2[i]*x[2])**2
        return y

class st_cqpjk2(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 0.9], [0, 0.9], [0, 0.9]]
        
    def _function(self,x):
        return 9*x[0]*x[0]-15*x[0]+9*x[1]*x[1]-12*x[1]+9*x[2]*x[2]-9*x[2]

# needs more precise floats
class yfit(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9978786299, 10.0021213701],
                [-10.0035439984, 9.9964560016], [0, 10010]]

    def _function(self,x):
        coeff1 = [x*0.0625 for x in range(17)]
        coeff0 = [x*0.0625 for x in reversed(range(17))]
        coeff_s = [-21.158931,-17.591719,-14.046854,-10.519732,-7.0058392,
                   -3.5007293,0,3.5007293,7.0058392,10.519732,14.046854,
                   17.591719,21.158931,24.753206,28.379405,32.042552,35.747869]
        return numpy.sum((coeff_s+
                          numpy.arctan(numpy.multiply(coeff0,x[0])+
                                       numpy.multiply(coeff1,x[1]))*x[2])**2)

# new
class allinitu(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.5401384356, 10.31387540796], [-10.0, 9.0], [-9.9191099435, 9.07280105085], [-10.8111130846, 8.26999822386]]

    def _function(self,x):
        v = numpy.zeros(5)
        v[0] = x[0] * x[0]
        v[1] = x[1] * x[1]
        v[0] += v[1]
        v[1] = x[2] + x[3]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = numpy.sin(x[2]);
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = x[0] * x[0]
        v[2] = x[1] * x[1]
        v[3] = v[1] * v[2]
        v[0] += v[3]
        v[3] = numpy.sin(x[2]);
        v[1] = v[3] * v[3]
        v[0] += v[1]
        v[1] = -1. + x[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = x[1] * x[1]
        v[1] = v[3] * v[3]
        v[0] += v[1]
        v[1] = x[2] * x[2]
        v[3] = x[3] + x[0]
        v[2] = v[3] * v[3]
        v[3] = v[1] + v[2]
        v[1] = v[3] * v[3]
        v[0] += v[1]
        v[1] = -4. + x[0]
        v[3] = numpy.sin(x[3]);
        v[2] = v[3] * v[3]
        v[1] += v[2]
        v[2] = x[1] * x[1]
        v[3] = x[2] * x[2]
        v[4] = v[2] * v[3]
        v[1] += v[4]
        v[4] = v[1] * v[1]
        v[0] += v[4]
        v[4] = numpy.sin(x[3]);
        v[1] = pow(v[4], 4)
        v[0] += v[1]
        v[0] += -4.;
        rv = v[0] + x[2]
        rv += x[3]
        return rv;

# needs more precise floats
class brownden(BenchmarkFunction):
    @property
    def domain(self):
        return [[-21.5944399048, -1.43499591432],
                [3.2036300512, 20.88326704608],
                [-10.4034394882, 8.63690446062],
                [-9.7632212255, 9.21310089705]]
    def _function(self,x):
        coeff1 = [round(x*0.2,ndigits=1) for x in range(1,21)]
        coeff_s1 = [-1.22140275816017,-1.49182469764127,-1.82211880039051,
                    -2.22554092849247,-2.71828182845905,-3.32011692273655,
                    -4.05519996684467,-4.95303242439511,-6.04964746441295,
                    -7.38905609893065,-9.02501349943412,-11.0231763806416,
                    -13.4637380350017,-16.444646771097,-20.0855369231877,
                    -24.5325301971094,-29.964100047397,-36.598234443678,
                    -44.7011844933008,-54.5981500331442]
        coeff_s2 = [-0.980066577841242,-0.921060994002885,-0.825335614909678,
                    -0.696706709347165,-0.54030230586814,-0.362357754476674,
                    -0.169967142900241,0.0291995223012888,0.227202094693087,
                    0.416146836547142,0.588501117255346,0.737393715541245,
                    0.856888753368947,0.942222340668658,0.989992496600445,
                    0.998294775794753,0.966798192579461,0.896758416334147,
                    0.790967711914417,0.653643620863612]
        coeff3 = [0.198669330795061,0.389418342308651,0.564642473395035,
                  0.717356090899523,0.841470984807897,0.932039085967226,
                  0.98544972998846,0.999573603041505,0.973847630878195,
                  0.909297426825682,0.80849640381959,0.675463180551151,
                  0.515501371821464,0.334988150155905,0.141120008059867,
                  -0.0583741434275801,-0.255541102026831,-0.442520443294852,
                  -0.611857890942719,-0.756802495307928]
        y = numpy.sum(((coeff_s1 + x[0] + numpy.multiply(coeff1,x[1]))**2 +
                       (coeff_s2 + x[2] + numpy.multiply(coeff3,x[3]))**2)**2)
        return y

class _hatfld(BenchmarkFunction):
    def _function(self,x):
        return ((x[0] - 1)**2 + (x[0] - numpy.sqrt(x[1]))**2 +
                (x[1] - numpy.sqrt(x[2]))**2 + (x[2] - numpy.sqrt(x[3]))**2)

class hatflda(_hatfld):
    @property
    def domain(self):
        return [[1e-07, 10.999999997], [1e-07, 10.9999999714],
                [1e-07, 10.9999999281], [1e-07, 10.9999998559]]

class hatfldb(_hatfld):
    @property
    def domain(self):
        return [[1e-07, 10.9472135922], [1e-07, 0.8],
                [1e-07, 10.6400000036], [1e-07, 10.4096000079]]

class hatfldc(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10], [-8.9999999978, 11.0000000022]]

    def _function(self,x):
        return (x[0]-1)**2 + (x[2]-x[1]**2)**2 + (x[3]-x[2]**2)**2 + (x[3]-1)**2

class himmelbf(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 0.378], [0, 0.378], [0, 0.378], [0, 0.378]]

    def _function(self,x):
        coeff0 = numpy.array([0.135299688810716,1,1,1,1,1,1])
        coeff1 = numpy.array([0,4.28e-4,1e-3,1.61e-3,2.09e-3,3.48e-3,0.00525])
        coeff2 = numpy.array([0,1.83184e-7,1e-6,2.5921e-6,4.3681e-6,
                              1.21104e-5,2.75625e-5])
        coeff3 = numpy.array([0,4.78504e-3,0.01644,0.026082,0.046398,
                              0.0835896,0.16443])
        coeffa = numpy.array([1,11.18,16.44,16.2,22.2,24.02,31.32])
        return numpy.sum(((coeff0*x[0]**2+coeff1*x[1]**2+
                           coeff2*x[2]**2)/(coeffa+coeff3*x[3]**2)-1)**2)*1e4

# new
class hs038(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0, 10.0], [-9.0, 11.0], [-9.0000000001, 10.9999999999], [-9.0000000001, 10.9999999999]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[3] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 90. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = v[2] * v[2]
        v[2] = 10.1 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[3]
        v[0] = v[2] * v[2]
        v[2] = 10.1 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = 19.8 * v[2]
        v[2] = -1. + x[3]
        v[3] = v[0] * v[2]
        v[1] += v[3]
        return v[1]

class kowalik(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 0.378], [0, 0.378], [0, 0.378], [0, 0.378]]

    def _function(self,x):
        term1 = [0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,
                 0.0235,0.0246]
        term2 = [16,4,1,0.25,0.0625,0.0277777777777778,0.015625,0.01,
                 0.00694444444444444,0.00510204081632653,0.00390625]
        coeff = [4,2,1,0.5,0.25,0.166666666666667,0.125,0.1,0.0833333333333333,
                 0.0714285714285714,0.0625]
        return numpy.sum((term1 - x[0]*(term2 + numpy.multiply(coeff,x[1]))/
                          (term2 + numpy.multiply(coeff,x[2]) + x[3]))**2)

# new
class kowosb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.8071930634, 9.17352624294], [-9.8087176971, 9.17215407261], [-9.8769435657, 9.11075079087], [-9.8639376421, 9.12245612211]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 4. * x[1]
        v[1] = 16. + v[0]
        v[0] = x[0] * v[1]
        v[1] = 4. * x[2]
        v[1] += 16.;
        v[1] += x[3]
        if (v[1] == 0.):
            v[2] = v[0] / v[1]
        else:
            v[2] = v[0]
        v[0] = 0.1957 - v[2]
        v[2] = v[0] * v[0]
        v[0] = 2. * x[1]
        v[1] = 4. + v[0]
        v[0] = x[0] * v[1]
        v[1] = 2. * x[2]
        v[1] += 4.;
        v[1] += x[3]
        if (v[1] == 0.):
            v[3] = v[0] / v[1]
        else:
            v[3] = v[0]
        v[0] = 0.1947 - v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 1. + x[1]
        v[0] = x[0] * v[3]
        v[3] = 1. + x[2]
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.1735 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.5 * x[1]
        v[0] = 0.25 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.5 * x[2]
        v[0] += 0.25;
        v[0] += x[3]
        if (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.16 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.25 * x[1]
        v[1] = 0.0625 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.25 * x[2]
        v[1] += 0.0625;
        v[1] += x[3]
        if (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0844 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        v[0] = 0.167 * x[1]
        v[3] = 0.027889000000000004 + v[0]
        v[0] = x[0] * v[3]
        v[3] = 0.167 * x[2]
        v[3] += 0.027889000000000004;
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.0627 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.125 * x[1]
        v[0] = 0.015625 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.125 * x[2]
        v[0] += 0.015625;
        v[0] += x[3]
        if (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.0456 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.1 * x[1]
        v[1] = 0.010000000000000002 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.1 * x[2]
        v[1] += 0.010000000000000002;
        v[1] += x[3]
        if (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0342 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        v[0] = 0.0833 * x[1]
        v[3] = 0.00693889 + v[0]
        v[0] = x[0] * v[3]
        v[3] = 0.0833 * x[2]
        v[3] += 0.00693889;
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.0323 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.0714 * x[1]
        v[0] = 0.00509796 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.0714 * x[2]
        v[0] += 0.00509796;
        v[0] += x[3]
        if not (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.0235 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.0625 * x[1]
        v[1] = 0.00390625 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.0625 * x[2]
        v[1] += 0.00390625;
        v[1] += x[3]
        if not (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0246 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        return v[2]

# needs more precise floats
class palmer1(BenchmarkFunction):
    @property
    def domain(self):
        return [[1.3636340716, 21.3636340716], [1e-05, 160.4544000091],
                [1e-05, 11.5013647921], [1e-05, 10.0931561774]]

    def _function(self,x):
        term1 = [78.596218,65.77963,43.96947,27.038816,14.6126,6.2614,1.53833,
                 0,1.188045,4.6841,16.9321,33.6988,52.3664,70.163,83.4221,
                 88.3995,78.596218,65.77963,43.96947,27.038816,14.6126,6.2614,
                 1.53833,0,1.188045,4.6841,16.9321,33.6988,52.3664,70.163,
                 83.4221]
        term2 = [3.200388615369,3.046173318241,2.749172911969,2.467400073616,
                 2.2008612609,1.949550365169,1.713473146009,1.485015206544,
                 1.287008567296,1.096623651204,0.761544202225,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089,0,3.200388615369,
                 3.046173318241,2.749172911969,2.467400073616,2.2008612609,
                 1.949550365169,1.713473146009,1.485015206544,1.287008567296,
                 1.096623651204,0.761544202225,0.487388289424,0.274155912801,
                 0.121847072356,0.030461768089]
        return numpy.sum((term1 - x[1]/(term2/x[3] + x[2]) -
                          numpy.multiply(term2,x[0]))**2)

# needs more precise floats
class palmer3(BenchmarkFunction):
    @property
    def domain(self):
        return [[1e-06, 10.0375049888], [1e-06, 10.0034428969],
                [1e-06, 14.6439962785], [7.3225711014, 27.3225711014]]

    def _function(self,x):
        term1 = [64.87939,50.46046,28.2034,13.4575,4.6547,0.59447,0,0.2177,
                 2.3029,5.5191,8.5519,9.8919,8.5519,5.5191,2.3029,0.2177,
                 0,0.59447,4.6547,13.4575,28.2034,50.46046,64.87939]
        term2 = [2.749172911969,2.467400073616,1.949550365169,1.4926241929,
                 1.096623651204,0.761544202225,0.587569773961,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089,0,0.030461768089,
                 0.121847072356,0.274155912801,0.487388289424,0.587569773961,
                 0.761544202225,1.096623651204,1.4926241929,1.949550365169,
                 2.467400073616,2.749172911969]
        return numpy.sum((term1 - x[0]/(term2/x[2] + x[1]) -
                          numpy.multiply(term2,x[3]))**2)


# needs more precise floats
class palmer4(BenchmarkFunction):
    @property
    def domain(self):
        return [[1e-05, 19.3292787916], [1e-05, 10.8767116668],
                [1e-05, 10.0158603779], [8.2655580306, 28.2655580306]]

    def _function(self,x):
        term1 = [67.27625,52.8537,30.2718,14.9888,5.56750,0.92603,0,0.085108,
                 1.867422,5.014768,8.263520,9.8046208,8.263520,5.014768,
                 1.867422,0.085108,0,0.92603,5.5675,14.9888,30.2718,52.8537,
                 67.27625]
        term2 = [2.749172911969,2.467400073616,1.949550365169,1.4926241929,
                 1.096623651204,0.761544202225,0.549257372161,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089,0,0.030461768089,
                 0.121847072356,0.274155912801,0.487388289424,0.549257372161,
                 0.761544202225,1.096623651204,1.4926241929,1.949550365169,
                 2.467400073616,2.749172911969]
        return numpy.sum((term1 - x[0]/(term2/x[2] + x[1]) -
                          numpy.multiply(term2,x[3]))**2)

# needs more precise floats
class palmer5d(BenchmarkFunction):
    @property
    def domain(self):
        return [[70.2513178169, 81.22618603521], [-142.1059487487, -109.89535387383],
                [41.6401308813, 55.47611779317], [-9.304685674, 9.6257828934]]

    def _function(self,x):
        term1 = [83.57418,81.007654,18.983286,8.051067,2.044762,0,1.170451,
                 10.479881,25.785001,44.126844,62.822177,77.719674]
        term2 = [0,2.467400073616,1.949550365169,1.713473146009,1.4926241929,
                 1.267504447225,1.096623651204,0.761544202225,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089]
        term3 = [0,6.08806312328024,3.80074662633058,2.93599022209398,
                 2.22792698123038,1.60656752373515,1.20258343237999,
                 0.579949571942512,0.237547344667653,0.0751614645237495,
                 0.0148467090417283,9.27919315108019e-4]
        term4 = [0,15.0216873985605,7.40974697327763,5.03074040250303,
                 3.32545771219912,2.03633148110156,1.31878143449399,
                 0.44165723409569,0.115777793974781,0.0206059599139685,
                 0.00180902803085595,2.82660629821242e-5]
        return numpy.sum((term1 - x[0] - numpy.multiply(term2,x[1]) -
                          numpy.multiply(term3,x[2])-
                          numpy.multiply(term4,x[3]))**2)

# new
class powell(BenchmarkFunction):
    @property
    def domain(self):
        return [[-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 10. * x[1]
        v[1] = x[0] + v[0]
        v[0] = v[1] * v[1]
        v[1] = x[2] - x[3]
        v[2] = v[1] * v[1]
        v[1] = 5. * v[2]
        v[0] += v[1]
        v[1] = -2. * x[2]
        v[2] = x[1] + v[1]
        v[1] = pow(v[2], 4)
        v[0] += v[1]
        v[1] = x[0] - x[3]
        v[2] = pow(v[1], 4)
        v[1] = 10. * v[2]
        v[0] += v[1]
        return v[0]
# new
class pspdoc(BenchmarkFunction):
    @property
    def domain(self):
        return [[-11.0, 0.0], [-9.999999972, 10.000000028], [-9.9999999213, 10.0000000787], [-9.9999998676, 10.0000001324]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[0] += 1.;
        v[1] = x[1] - x[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = numpy.sqrt(v[0]);
        v[0] = x[1] * x[1]
        v[0] += 1.;
        v[1] = x[2] - x[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = numpy.sqrt(v[0]);
        v[0] = v[2] + v[3]
        return v[0]
# new
class s256(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9981379886, 9.00167581026], [-10.0001862011, 8.99983241901], [-9.9994511045, 9.00049400595], [-9.9994511048, 9.00049400568]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 10. * x[1]
        v[1] = x[0] + v[0]
        v[0] = v[1] * v[1]
        v[1] = x[2] - x[3]
        v[2] = v[1] * v[1]
        v[1] = 5. * v[2]
        v[0] += v[1]
        v[1] = -2. * x[2]
        v[2] = x[1] + v[1]
        v[1] = pow(v[2], 4)
        v[0] += v[1]
        v[1] = x[0] - x[3]
        v[2] = pow(v[1], 4)
        v[1] = 10. * v[2]
        v[0] += v[1]
        return v[0]

class s257(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 11], [-9, 11], [0, 11], [-9, 11]]

    def _function(self,x):
        return (100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 +
                90*(x[2]**2 - x[3])**2 + (x[2] - 1)**2 +
                10.1*((x[1] - 1)**2 + (x[3] - 1)**2)+(19.8*x[0]-19.8)*(x[3]-1))

# new
class s258(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-8.9999999999, 9.90000000009]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[3] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 90. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = v[2] * v[2]
        v[2] = 10.1 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[3]
        v[0] = v[2] * v[2]
        v[2] = 10.1 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = 19.8 * v[2]
        v[2] = -1. + x[3]
        v[3] = v[0] * v[2]
        v[1] += v[3]
        return v[1]
# new
class s259(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.5641580904, 11.4358419096], [-7.936840288, 12.063159712], [-9.9310254894, 10.0689745106], [-10.0999682161, 0.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[3] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 90. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[2]
        v[2] = pow(v[0], 3)
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = v[2] * v[2]
        v[2] = 10.1 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = -1. + x[1]
        v[2] = 19.8 * v[0]
        v[0] = -1. + x[3]
        v[3] = v[2] * v[0]
        v[1] += v[3]
        return v[1]
# new
class s260(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-8.9999999999, 9.90000000009]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[3] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 90. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = -2. + x[1]
        v[2] += x[3]
        v[0] = v[2] * v[2]
        v[2] = 9.9 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[1]
        v[0] = v[2] * v[2]
        v[2] = 0.2 * v[0]
        v[1] += v[2]
        v[2] = -1. + x[3]
        v[0] = v[2] * v[2]
        v[2] = 0.2 * v[0]
        v[1] += v[2]
        return v[1]
        
# new
class s261(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9909450357, 9.00814946787], [-8.991563976, 9.9075924216], [-9.0016688312, 9.89849805192], [-9.0000000002, 9.89999999982]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = numpy.exp(x[0]);
        v[1] = v[0] - x[1]
        v[0] = pow(v[1], 4)
        v[1] = x[1] - x[2]
        v[2] = pow(v[1], 6)
        v[1] = 100. * v[2]
        v[0] += v[1]
        v[1] = x[2] - x[3]
        v[2] = numpy.tan(v[1]);
        v[1] = pow(v[2], 4)
        v[0] += v[1]
        v[1] = pow(x[0], 8)
        v[0] += v[1]
        v[1] = -1. + x[3]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        return v[0]
# new
class s275(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9999927549, 9.00000652059], [-10.000081616, 8.9999265456], [-9.999803482, 9.0001768662], [-10.0001277807, 8.99988499737]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 0.5 * x[1]
        v[0] += x[0]
        v[1] = 0.3333333333333333 * x[2]
        v[0] += v[1]
        v[1] = 0.25 * x[3]
        v[0] += v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.5 * x[0]
        v[2] = 0.3333333333333333 * x[1]
        v[0] += v[2]
        v[2] = 0.25 * x[2]
        v[0] += v[2]
        v[2] = 0.2 * x[3]
        v[0] += v[2]
        v[2] = x[1] * v[0]
        v[1] += v[2]
        v[2] = 0.3333333333333333 * x[0]
        v[0] = 0.25 * x[1]
        v[2] += v[0]
        v[0] = 0.2 * x[2]
        v[2] += v[0]
        v[0] = 0.16666666666666666 * x[3]
        v[2] += v[0]
        v[0] = x[2] * v[2]
        v[1] += v[0]
        v[0] = 0.25 * x[0]
        v[2] = 0.2 * x[1]
        v[0] += v[2]
        v[2] = 0.16666666666666666 * x[2]
        v[0] += v[2]
        v[2] = 0.14285714285714285 * x[3]
        v[0] += v[2]
        v[2] = x[3] * v[0]
        v[1] += v[2]
        return v[1]
# new
class s350(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.75, 9.225], [-9.61, 9.351], [-9.585, 9.3735], [-9.61, 9.351]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 4. * x[1]
        v[1] = 16. + v[0]
        v[0] = x[0] * v[1]
        v[1] = 4. * x[2]
        v[1] += 16.;
        v[1] += x[3]
        if (v[1] == 0):
            v[2] = v[0] / v[1]
        else:
            v[2] = v[0]
        v[0] = 0.1957 - v[2]
        v[2] = v[0] * v[0]
        v[0] = 2. * x[1]
        v[1] = 4. + v[0]
        v[0] = x[0] * v[1]
        v[1] = 2. * x[2]
        v[1] += 4.;
        v[1] += x[3]
        if (v[1] == 0):
            v[3] = v[0] / v[1]
        else:
            v[3] = v[0]
        v[0] = 0.1947 - v[3]
        v[3] = v[0] * v[0]
        v[2] += v[3]
        v[3] = 1. + x[1]
        v[0] = x[0] * v[3]
        v[3] = 1. + x[2]
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.1735 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.5 * x[1]
        v[0] = 0.25 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.5 * x[2]
        v[0] += 0.25;
        v[0] += x[3]
        if (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.16 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.25 * x[1]
        v[1] = 0.0625 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.25 * x[2]
        v[1] += 0.0625;
        v[1] += x[3]
        if (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0844 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        v[0] = 0.167 * x[1]
        v[3] = 0.027889000000000004 + v[0]
        v[0] = x[0] * v[3]
        v[3] = 0.167 * x[2]
        v[3] += 0.027889000000000004;
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.0627 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.125 * x[1]
        v[0] = 0.015625 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.125 * x[2]
        v[0] += 0.015625;
        v[0] += x[3]
        if (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.0456 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.1 * x[1]
        v[1] = 0.010000000000000002 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.1 * x[2]
        v[1] += 0.010000000000000002;
        v[1] += x[3]
        if (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0342 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        v[0] = 0.0833 * x[1]
        v[3] = 0.00693889 + v[0]
        v[0] = x[0] * v[3]
        v[3] = 0.0833 * x[2]
        v[3] += 0.00693889;
        v[3] += x[3]
        if (v[3] == 0):
            v[1] = v[0] / v[3]
        else:
            v[1] = v[0]
        v[0] = 0.0323 - v[1]
        v[1] = v[0] * v[0]
        v[2] += v[1]
        v[1] = 0.0714 * x[1]
        v[0] = 0.00509796 + v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.0714 * x[2]
        v[0] += 0.00509796;
        v[0] += x[3]
        if (v[0] == 0):
            v[3] = v[1] / v[0]
        else:
            v[3] = v[1]
        v[1] = 0.0235 - v[3]
        v[3] = v[1] * v[1]
        v[2] += v[3]
        v[3] = 0.0625 * x[1]
        v[1] = 0.00390625 + v[3]
        v[3] = x[0] * v[1]
        v[1] = 0.0625 * x[2]
        v[1] += 0.00390625;
        v[1] += x[3]
        if (v[1] == 0):
            v[0] = v[3] / v[1]
        else:
            v[0] = v[3]
        v[3] = 0.0246 - v[0]
        v[0] = v[3] * v[3]
        v[2] += v[0]
        return v[2]

class s351(BenchmarkFunction):
    @property
    def domain(self):
        return [[-7.3, 11.43], [80, 90], [1359, 1490], [0, 18]]

    def _function(self,x):
        term0 = numpy.array([0.135299688810716,0.0894454382826476,
                             0.0608272506082725,0.0617283950617284,
                             0.045045045045045,0.0416319733555371,
                             0.0416319733555371],dtype=numpy.longdouble)
        term1 = numpy.array([0,4.28e-4,0.001,0.00161,0.00209,0.00348,0.00525])
        term2 = numpy.array([0,1.83184e-7,1e-6,2.5921e-6,4.3681e-6,1.21104e-5,
                             2.75625e-5])
        return numpy.sum((term0*(x[0]**2 +term1*x[1]**2 +
                                 term2*x[2]**2)/(1+term1*x[3]**2)-1)**2)*1e4

# needs more precise floats
class s352(BenchmarkFunction):
    @property
    def domain(self):
        return [[-20.2235736001, -0.20121624009],
                [1.9084286837, 19.71758581533],
                [-10.4580411955, 8.58776292405],
                [-9.4196803043, 9.52228772613]]

    def _function(self,x):
        term1 = [-1.22140275816017,-1.49182469764127,-1.82211880039051,
                 -2.22554092849247,-2.71828182845905,-3.32011692273655,
                 -4.05519996684468,-4.95303242439511,-6.04964746441295,
                 -7.38905609893065,-9.02501349943412,-11.0231763806416,
                 -13.4637380350017,-16.4446467710971,-20.0855369231877,
                 -24.5325301971094,-29.964100047397,-36.598234443678,
                 -44.7011844933008,-54.5981500331442]
        term2 = [round(x*0.2,ndigits=1) for x in range(1,21)]
        term3 = [-0.980066577841242,-0.921060994002885,-0.825335614909678,
                 -0.696706709347165,-0.54030230586814,-0.362357754476673,
                 -0.169967142900241,0.0291995223012888,0.227202094693087,
                 0.416146836547142,0.588501117255346,0.737393715541246,
                 0.856888753368947,0.942222340668658,0.989992496600445,
                 0.998294775794753,0.966798192579461,0.896758416334147,
                 0.790967711914417,0.653643620863612]
        term4 = [0.198669330795061,0.389418342308651,0.564642473395035,
                 0.717356090899523,0.841470984807897,0.932039085967226,
                 0.98544972998846,0.999573603041505,0.973847630878195,
                 0.909297426825682,0.80849640381959,0.675463180551151,
                 0.515501371821464,0.334988150155905,0.141120008059867,
                 -0.0583741434275801,-0.255541102026832,-0.442520443294852,
                 -0.611857890942719,-0.756802495307928]
        return numpy.sum((term1 + x[0] + numpy.multiply(term2,x[1]))**2 +
                         (term3 + x[2] + numpy.multiply(term4,x[3]))**2)

class shekel(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10], [0, 10], [0, 10], [0, 10]]
    def _function(self,x):
        term1 = [0.1,0.2,0.2,0.4,0.4]
        term2 = [4,1,8,6,3]
        term3 = [4,1,8,6,7]
        y = -numpy.sum(1/(term1 + (x[0] - term2)**2 + (x[1] - term3)**2 +
                          (x[2] - term2)**2 + (x[3] - term3)**2))
        return y

# new
class aircrftb(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.9654790084, 8.13106889244], [-9.9364470184, 9.05719768344], [-10.0579260446, 8.94786655986], [-9.9385332696, 9.05532005736], [-10.0108611902, 8.99022492882]]

    def _function(self,x):
        v = numpy.zeros(18)
        v[8] = -4.583 - 3.933*x[0]
        v[8] += 0.107*x[1]
        v[8] += 0.126*x[2]
        v[8] -= 9.99*x[4]

        v[9] = 1.4185 - 0.987*x[1]
        v[9] -= 22.95*x[3]

        v[10] = -0.09210000000000002 + 0.002*x[0]
        v[10] -= 0.235*x[2]
        v[10] += 5.67*x[4]

        v[11] = 0.008400000000000001 + x[1]
        v[11] -= x[3]

        v[12] = -0.0007100000000000001 - x[2]
        v[12] -= 0.196*x[4]

        v[0] = -0.727 * x[1]
        v[1] = v[0] * x[2]
        v[0] = 8.39 * x[2]
        v[2] = v[0] * x[3]
        v[13] = v[1] + v[2]
        v[2] = 684.4 * x[3]
        v[0] = v[2] * x[4]
        v[2] = -v[0]
        v[13] += v[2]
        v[2] = 63.5 * x[3]
        v[0] = v[2] * x[1]
        v[13] += v[0]

        v[0] = 0.949 * x[0]
        v[2] = v[0] * x[2]
        v[0] = 0.173 * x[0]
        v[3] = v[0] * x[4]
        v[14] = v[2] + v[3]

        v[2] = -0.716 * x[0]
        v[3] = v[2] * x[1]
        v[2] = 1.578 * x[0]
        v[4] = v[2] * x[3]
        v[2] = -v[4]
        v[15] = v[3] + v[2]
        v[2] = 1.132 * x[3]
        v[4] = v[2] * x[1]
        v[15] += v[4]

        v[4] = x[0] * x[4]
        v[16] = -v[4]
        
        v[17] = x[0] * x[3]

        v[5] = v[8] + v[13]
        v[6] = v[5] * v[5]
        v[5] = v[9] + v[14]
        v[7] = v[5] * v[5]
        v[6] += v[7]
        v[7] = v[10] + v[15]
        v[5] = v[7] * v[7]
        v[6] += v[5]
        v[5] = v[11] + v[16]
        v[7] = v[5] * v[5]
        v[6] += v[7]
        v[7] = v[12] + v[17]
        v[5] = v[7] * v[7]
        v[6] += v[5]
        return v[6]
# new
class biggs5(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.9999996463, 9.90000031833], [-9.10100000695024e-07, 17.99999918091], [-8.999999498, 9.9000004518], [-4.999999779, 13.5000001989], [-5.999999158, 12.6000007578]]

    def _function(self,x):
        v = numpy.zeros(3)

# new
class genhumps(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.999999999, 9.0000000009], [-10.0000000017, 8.99999999847], [-10.0000000095, 8.99999999145], [-9.9999999989, 9.00000000099], [-10.0000000027, 8.99999999757]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 2. * x[0]
        v[1] = numpy.sin(v[0]);
        v[0] = v[1] * v[1]
        v[1] = 2. * x[1]
        v[2] = numpy.sin(v[1]);
        v[1] = v[2] * v[2]
        v[2] = v[0] * v[1]
        v[0] = x[0] * x[0]
        v[1] = 0.05 * v[0]
        v[2] += v[1]
        v[1] = x[1] * x[1]
        v[0] = 0.05 * v[1]
        v[2] += v[0]
        v[0] = 2. * x[1]
        v[1] = numpy.sin(v[0]);
        v[0] = v[1] * v[1]
        v[1] = 2. * x[2]
        v[3] = numpy.sin(v[1]);
        v[1] = v[3] * v[3]
        v[3] = v[0] * v[1]
        v[2] += v[3]
        v[3] = x[1] * x[1]
        v[0] = 0.05 * v[3]
        v[2] += v[0]
        v[0] = x[2] * x[2]
        v[3] = 0.05 * v[0]
        v[2] += v[3]
        v[3] = 2. * x[2]
        v[0] = numpy.sin(v[3]);
        v[3] = v[0] * v[0]
        v[0] = 2. * x[3]
        v[1] = numpy.sin(v[0]);
        v[0] = v[1] * v[1]
        v[1] = v[3] * v[0]
        v[2] += v[1]
        v[1] = x[2] * x[2]
        v[3] = 0.05 * v[1]
        v[2] += v[3]
        v[3] = x[3] * x[3]
        v[1] = 0.05 * v[3]
        v[2] += v[1]
        v[1] = 2. * x[3]
        v[3] = numpy.sin(v[1]);
        v[1] = v[3] * v[3]
        v[3] = 2. * x[4]
        v[0] = numpy.sin(v[3]);
        v[3] = v[0] * v[0]
        v[0] = v[1] * v[3]
        v[2] += v[0]
        v[0] = x[3] * x[3]
        v[1] = 0.05 * v[0]
        v[2] += v[1]
        v[1] = x[4] * x[4]
        v[0] = 0.05 * v[1]
        v[2] += v[0]
        return v[2]

class hs045(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]

    def _function(self,x):
        return -(0.00833333333333333*x[0]*x[1]*x[2]*x[3]*x[4] - 2)

# new
class osborne1(BenchmarkFunction):
    @property
    def domain(self):
        # originally -2 and 2 for all, changed to avoid runtime warning
        return [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                [-1.0, 2.0], [-0.5, 0.5]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] + x[1]
        v[0] += x[2]
        v[1] = 0.844 - v[0]
        v[0] = v[1] * v[1]
        v[1] = 10. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 10. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.908 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 20. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 20. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.932 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 30. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 30. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.936 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 40. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 40. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.925 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 50. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 50. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.908 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 60. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 60. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.881 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 70. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 70. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.85 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 80. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 80. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.818 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 90. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 90. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.784 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 100. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 100. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.751 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 110. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 110. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.718 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 120. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 120. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.685 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 130. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 130. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.658 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 140. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 140. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.628 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 150. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 150. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.603 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 160. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 160. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.58 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 170. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 170. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.558 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 180. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 180. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.538 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 190. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 190. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.522 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 200. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 200. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.506 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 210. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 210. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.49 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 220. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 220. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.478 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 230. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 230. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.467 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 240. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 240. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.457 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 250. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 250. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.448 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 260. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 260. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.438 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 270. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 270. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.431 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 280. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 280. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.424 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 290. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 290. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.42 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        v[3] = 300. * x[3]
        v[1] = -v[3]
        v[3] = numpy.exp(v[1]);
        v[1] = x[1] * v[3]
        v[1] += x[0]
        v[3] = 300. * x[4]
        v[2] = -v[3]
        v[3] = numpy.exp(v[2]);
        v[2] = x[2] * v[3]
        v[1] += v[2]
        v[2] = 0.414 - v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 310. * x[3]
        v[2] = -v[1]
        v[1] = numpy.exp(v[2]);
        v[2] = x[1] * v[1]
        v[2] += x[0]
        v[1] = 310. * x[4]
        v[3] = -v[1]
        v[1] = numpy.exp(v[3]);
        v[3] = x[2] * v[1]
        v[2] += v[3]
        v[3] = 0.411 - v[2]
        v[2] = v[3] * v[3]
        v[0] += v[2]
        v[2] = 320. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[3] += x[0]
        v[2] = 320. * x[4]
        v[1] = -v[2]
        v[2] = numpy.exp(v[1]);
        v[1] = x[2] * v[2]
        v[3] += v[1]
        v[1] = 0.406 - v[3]
        v[3] = v[1] * v[1]
        v[0] += v[3]
        return v[0]
# new
class osbornea(BenchmarkFunction):
    @property
    def domain(self):
        return [[-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = 0.844 - x[0]
        v[1] = v[0] - x[1]
        v[0] = v[1] - x[2]
        v[1] = v[0] * v[0]
        v[0] = 0.908 - x[0]
        v[2] = 10. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 10. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.932 - x[0]
        v[0] = 20. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 20. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.936 - x[0]
        v[2] = 30. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 30. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.925 - x[0]
        v[0] = 40. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 40. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.908 - x[0]
        v[2] = 50. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 50. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.881 - x[0]
        v[0] = 60. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 60. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.85 - x[0]
        v[2] = 70. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 70. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.818 - x[0]
        v[0] = 80. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 80. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.784 - x[0]
        v[2] = 90. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 90. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.751 - x[0]
        v[0] = 100. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 100. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.718 - x[0]
        v[2] = 110. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 110. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.685 - x[0]
        v[0] = 120. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 120. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.658 - x[0]
        v[2] = 130. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 130. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.628 - x[0]
        v[0] = 140. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 140. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.603 - x[0]
        v[2] = 150. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 150. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.58 - x[0]
        v[0] = 160. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 160. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.558 - x[0]
        v[2] = 170. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 170. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.538 - x[0]
        v[0] = 180. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 180. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.522 - x[0]
        v[2] = 190. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 190. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.506 - x[0]
        v[0] = 200. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 200. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.49 - x[0]
        v[2] = 210. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 210. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.478 - x[0]
        v[0] = 220. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 220. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.467 - x[0]
        v[2] = 230. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 230. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.457 - x[0]
        v[0] = 240. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 240. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.448 - x[0]
        v[2] = 250. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 250. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.438 - x[0]
        v[0] = 260. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 260. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.431 - x[0]
        v[2] = 270. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 270. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.424 - x[0]
        v[0] = 280. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 280. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.42 - x[0]
        v[2] = 290. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 290. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.414 - x[0]
        v[0] = 300. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 300. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = 0.411 - x[0]
        v[2] = 310. * x[3]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[1] * v[2]
        v[2] = v[0] - v[3]
        v[0] = 310. * x[4]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[2] * v[0]
        v[0] = v[2] - v[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = 0.406 - x[0]
        v[0] = 320. * x[3]
        v[3] = -v[0]
        v[0] = numpy.exp(v[3]);
        v[3] = x[1] * v[0]
        v[0] = v[2] - v[3]
        v[2] = 320. * x[4]
        v[3] = -v[2]
        v[2] = numpy.exp(v[3]);
        v[3] = x[2] * v[2]
        v[2] = v[0] - v[3]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        return v[1]
# new
class s266(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9999999907, 9.00000000837], [-10.0000000009, 8.99999999919], [-10.0000000271, 8.99999997561], [-10.0, 9.0], [-9.9999999805, 9.00000001755]]

    def _function(self,x):
        v = numpy.zeros(24)
        v[0] = 0.354033 * x[0]
        v[1] = -0.0230349 * x[1]
        v[0] += v[1]
        v[1] = -0.211938 * x[2]
        v[0] += v[1]
        v[1] = -0.0554288 * x[3]
        v[0] += v[1]
        v[1] = 0.220429 * x[4]
        v[0] += v[1]
        v[1] = x[0] * v[0]
        v[0] = 1.173295 * v[1]
        v[1] = -0.0230349 * x[0]
        v[2] = 0.29135 * x[1]
        v[1] += v[2]
        v[2] = -0.00180333 * x[2]
        v[1] += v[2]
        v[2] = -0.111141 * x[3]
        v[1] += v[2]
        v[2] = 0.0485461 * x[4]
        v[1] += v[2]
        v[2] = x[1] * v[1]
        v[1] = 1.173295 * v[2]
        v[14] = v[0] + v[1]
        v[1] = -0.211938 * x[0]
        v[2] = -0.00180333 * x[1]
        v[1] += v[2]
        v[2] = 0.815808 * x[2]
        v[1] += v[2]
        v[2] = -0.133538 * x[3]
        v[1] += v[2]
        v[2] = -0.38067 * x[4]
        v[1] += v[2]
        v[2] = x[2] * v[1]
        v[1] = 1.173295 * v[2]
        v[14] += v[1]
        v[1] = -0.0554288 * x[0]
        v[2] = -0.111141 * x[1]
        v[1] += v[2]
        v[2] = -0.133538 * x[2]
        v[1] += v[2]
        v[2] = 0.389198 * x[3]
        v[1] += v[2]
        v[2] = -0.131586 * x[4]
        v[1] += v[2]
        v[2] = x[3] * v[1]
        v[1] = 1.173295 * v[2]
        v[14] += v[1]
        v[1] = 0.220429 * x[0]
        v[2] = 0.0485461 * x[1]
        v[1] += v[2]
        v[2] = -0.38067 * x[2]
        v[1] += v[2]
        v[2] = -0.131586 * x[3]
        v[1] += v[2]
        v[2] = 0.534706 * x[4]
        v[1] += v[2]
        v[2] = x[4] * v[1]
        v[1] = 1.173295 * v[2]
        v[14] += v[1]
        v[14] += 0.0426149;
        v[14] = v[14] - 0.564255*x[0]
        v[14] += 0.392417*x[1]
        v[14] -= 0.404979*x[2]
        v[14] += 0.927589*x[3]
        v[14] -= 0.0735084*x[4]

        v[1] = 0.354033 * x[0]
        v[2] = -0.0230349 * x[1]
        v[1] += v[2]
        v[2] = -0.211938 * x[2]
        v[1] += v[2]
        v[2] = -0.0554288 * x[3]
        v[1] += v[2]
        v[2] = 0.220429 * x[4]
        v[1] += v[2]
        v[2] = x[0] * v[1]
        v[1] = 1.42024 * v[2]
        v[2] = -0.0230349 * x[0]
        v[3] = 0.29135 * x[1]
        v[2] += v[3]
        v[3] = -0.00180333 * x[2]
        v[2] += v[3]
        v[3] = -0.111141 * x[3]
        v[2] += v[3]
        v[3] = 0.0485461 * x[4]
        v[2] += v[3]
        v[3] = x[1] * v[2]
        v[2] = 1.42024 * v[3]
        v[15] = v[1] + v[2]
        v[2] = -0.211938 * x[0]
        v[3] = -0.00180333 * x[1]
        v[2] += v[3]
        v[3] = 0.815808 * x[2]
        v[2] += v[3]
        v[3] = -0.133538 * x[3]
        v[2] += v[3]
        v[3] = -0.38067 * x[4]
        v[2] += v[3]
        v[3] = x[2] * v[2]
        v[2] = 1.42024 * v[3]
        v[15] += v[2]
        v[2] = -0.0554288 * x[0]
        v[3] = -0.111141 * x[1]
        v[2] += v[3]
        v[3] = -0.133538 * x[2]
        v[2] += v[3]
        v[3] = 0.389198 * x[3]
        v[2] += v[3]
        v[3] = -0.131586 * x[4]
        v[2] += v[3]
        v[3] = x[3] * v[2]
        v[2] = 1.42024 * v[3]
        v[15] += v[2]
        v[2] = 0.220429 * x[0]
        v[3] = 0.0485461 * x[1]
        v[2] += v[3]
        v[3] = -0.38067 * x[2]
        v[2] += v[3]
        v[3] = -0.131586 * x[3]
        v[2] += v[3]
        v[3] = 0.534706 * x[4]
        v[2] += v[3]
        v[3] = x[4] * v[2]
        v[2] = 1.42024 * v[3]
        v[15] += v[2]
        v[15] += 0.0352053;
        v[15] = v[15] + 0.535493*x[0]
        v[15] += 0.658799*x[1]
        v[15] -= 0.636666*x[2]
        v[15] -= 0.681091*x[3]
        v[15] -= 0.869487*x[4]

        v[2] = 0.354033 * x[0]
        v[3] = -0.0230349 * x[1]
        v[2] += v[3]
        v[3] = -0.211938 * x[2]
        v[2] += v[3]
        v[3] = -0.0554288 * x[3]
        v[2] += v[3]
        v[3] = 0.220429 * x[4]
        v[2] += v[3]
        v[3] = x[0] * v[2]
        v[2] = 0.56444 * v[3]
        v[3] = -0.0230349 * x[0]
        v[4] = 0.29135 * x[1]
        v[3] += v[4]
        v[4] = -0.00180333 * x[2]
        v[3] += v[4]
        v[4] = -0.111141 * x[3]
        v[3] += v[4]
        v[4] = 0.0485461 * x[4]
        v[3] += v[4]
        v[4] = x[1] * v[3]
        v[3] = 0.56444 * v[4]
        v[16] = v[2] + v[3]
        v[3] = -0.211938 * x[0]
        v[4] = -0.00180333 * x[1]
        v[3] += v[4]
        v[4] = 0.815808 * x[2]
        v[3] += v[4]
        v[4] = -0.133538 * x[3]
        v[3] += v[4]
        v[4] = -0.38067 * x[4]
        v[3] += v[4]
        v[4] = x[2] * v[3]
        v[3] = 0.56444 * v[4]
        v[16] += v[3]
        v[3] = -0.0554288 * x[0]
        v[4] = -0.111141 * x[1]
        v[3] += v[4]
        v[4] = -0.133538 * x[2]
        v[3] += v[4]
        v[4] = 0.389198 * x[3]
        v[3] += v[4]
        v[4] = -0.131586 * x[4]
        v[3] += v[4]
        v[4] = x[3] * v[3]
        v[3] = 0.56444 * v[4]
        v[16] += v[3]
        v[3] = 0.220429 * x[0]
        v[4] = 0.0485461 * x[1]
        v[3] += v[4]
        v[4] = -0.38067 * x[2]
        v[3] += v[4]
        v[4] = -0.131586 * x[3]
        v[3] += v[4]
        v[4] = 0.534706 * x[4]
        v[3] += v[4]
        v[4] = x[4] * v[3]
        v[3] = 0.56444 * v[4]
        v[16] += v[3]
        v[16] += 0.0878058;
        v[16] = v[16] + 0.586387*x[0]
        v[16] += 0.289826*x[1]
        v[16] += 0.854402*x[2]
        v[16] += 0.789312*x[3]
        v[16] += 0.949721*x[4]

        v[3] = 0.354033 * x[0]
        v[4] = -0.0230349 * x[1]
        v[3] += v[4]
        v[4] = -0.211938 * x[2]
        v[3] += v[4]
        v[4] = -0.0554288 * x[3]
        v[3] += v[4]
        v[4] = 0.220429 * x[4]
        v[3] += v[4]
        v[4] = x[0] * v[3]
        v[3] = 1.51143 * v[4]
        v[4] = -0.0230349 * x[0]
        v[5] = 0.29135 * x[1]
        v[4] += v[5]
        v[5] = -0.00180333 * x[2]
        v[4] += v[5]
        v[5] = -0.111141 * x[3]
        v[4] += v[5]
        v[5] = 0.0485461 * x[4]
        v[4] += v[5]
        v[5] = x[1] * v[4]
        v[4] = 1.51143 * v[5]
        v[17] = v[3] + v[4]
        v[4] = -0.211938 * x[0]
        v[5] = -0.00180333 * x[1]
        v[4] += v[5]
        v[5] = 0.815808 * x[2]
        v[4] += v[5]
        v[5] = -0.133538 * x[3]
        v[4] += v[5]
        v[5] = -0.38067 * x[4]
        v[4] += v[5]
        v[5] = x[2] * v[4]
        v[4] = 1.51143 * v[5]
        v[17] += v[4]
        v[4] = -0.0554288 * x[0]
        v[5] = -0.111141 * x[1]
        v[4] += v[5]
        v[5] = -0.133538 * x[2]
        v[4] += v[5]
        v[5] = 0.389198 * x[3]
        v[4] += v[5]
        v[5] = -0.131586 * x[4]
        v[4] += v[5]
        v[5] = x[3] * v[4]
        v[4] = 1.51143 * v[5]
        v[17] += v[4]
        v[4] = 0.220429 * x[0]
        v[5] = 0.0485461 * x[1]
        v[4] += v[5]
        v[5] = -0.38067 * x[2]
        v[4] += v[5]
        v[5] = -0.131586 * x[3]
        v[4] += v[5]
        v[5] = 0.534706 * x[4]
        v[4] += v[5]
        v[5] = x[4] * v[4]
        v[4] = 1.51143 * v[5]
        v[17] += v[4]
        v[17] += 0.0330812;
        v[17] = v[17] + 0.608734*x[0]
        v[17] += 0.984915*x[1]
        v[17] += 0.375699*x[2]
        v[17] += 0.239547*x[3]
        v[17] += 0.463136*x[4]

        v[4] = 0.354033 * x[0]
        v[5] = -0.0230349 * x[1]
        v[4] += v[5]
        v[5] = -0.211938 * x[2]
        v[4] += v[5]
        v[5] = -0.0554288 * x[3]
        v[4] += v[5]
        v[5] = 0.220429 * x[4]
        v[4] += v[5]
        v[5] = x[0] * v[4]
        v[4] = 0.860695 * v[5]
        v[5] = -0.0230349 * x[0]
        v[6] = 0.29135 * x[1]
        v[5] += v[6]
        v[6] = -0.00180333 * x[2]
        v[5] += v[6]
        v[6] = -0.111141 * x[3]
        v[5] += v[6]
        v[6] = 0.0485461 * x[4]
        v[5] += v[6]
        v[6] = x[1] * v[5]
        v[5] = 0.860695 * v[6]
        v[18] = v[4] + v[5]
        v[5] = -0.211938 * x[0]
        v[6] = -0.00180333 * x[1]
        v[5] += v[6]
        v[6] = 0.815808 * x[2]
        v[5] += v[6]
        v[6] = -0.133538 * x[3]
        v[5] += v[6]
        v[6] = -0.38067 * x[4]
        v[5] += v[6]
        v[6] = x[2] * v[5]
        v[5] = 0.860695 * v[6]
        v[18] += v[5]
        v[5] = -0.0554288 * x[0]
        v[6] = -0.111141 * x[1]
        v[5] += v[6]
        v[6] = -0.133538 * x[2]
        v[5] += v[6]
        v[6] = 0.389198 * x[3]
        v[5] += v[6]
        v[6] = -0.131586 * x[4]
        v[5] += v[6]
        v[6] = x[3] * v[5]
        v[5] = 0.860695 * v[6]
        v[18] += v[5]
        v[5] = 0.220429 * x[0]
        v[6] = 0.0485461 * x[1]
        v[5] += v[6]
        v[6] = -0.38067 * x[2]
        v[5] += v[6]
        v[6] = -0.131586 * x[3]
        v[5] += v[6]
        v[6] = 0.534706 * x[4]
        v[5] += v[6]
        v[6] = x[4] * v[5]
        v[5] = 0.860695 * v[6]
        v[18] += v[5]
        v[18] += 0.0580924;
        v[18] = v[18] + 0.774227*x[0]
        v[18] += 0.325421*x[1]
        v[18] -= 0.151719*x[2]
        v[18] += 0.448051*x[3]
        v[18] += 0.149926*x[4]

        v[5] = 0.354033 * x[0]
        v[6] = -0.0230349 * x[1]
        v[5] += v[6]
        v[6] = -0.211938 * x[2]
        v[5] += v[6]
        v[6] = -0.0554288 * x[3]
        v[5] += v[6]
        v[6] = 0.220429 * x[4]
        v[5] += v[6]
        v[6] = x[0] * v[5]
        v[5] = 0.0769585 * v[6]
        v[6] = -0.0230349 * x[0]
        v[7] = 0.29135 * x[1]
        v[6] += v[7]
        v[7] = -0.00180333 * x[2]
        v[6] += v[7]
        v[7] = -0.111141 * x[3]
        v[6] += v[7]
        v[7] = 0.0485461 * x[4]
        v[6] += v[7]
        v[7] = x[1] * v[6]
        v[6] = 0.0769585 * v[7]
        v[19] = v[5] + v[6]
        v[6] = -0.211938 * x[0]
        v[7] = -0.00180333 * x[1]
        v[6] += v[7]
        v[7] = 0.815808 * x[2]
        v[6] += v[7]
        v[7] = -0.133538 * x[3]
        v[6] += v[7]
        v[7] = -0.38067 * x[4]
        v[6] += v[7]
        v[7] = x[2] * v[6]
        v[6] = 0.0769585 * v[7]
        v[19] += v[6]
        v[6] = -0.0554288 * x[0]
        v[7] = -0.111141 * x[1]
        v[6] += v[7]
        v[7] = -0.133538 * x[2]
        v[6] += v[7]
        v[7] = 0.389198 * x[3]
        v[6] += v[7]
        v[7] = -0.131586 * x[4]
        v[6] += v[7]
        v[7] = x[3] * v[6]
        v[6] = 0.0769585 * v[7]
        v[19] += v[6]
        v[6] = 0.220429 * x[0]
        v[7] = 0.0485461 * x[1]
        v[6] += v[7]
        v[7] = -0.38067 * x[2]
        v[6] += v[7]
        v[7] = -0.131586 * x[3]
        v[6] += v[7]
        v[7] = 0.534706 * x[4]
        v[6] += v[7]
        v[7] = x[4] * v[6]
        v[6] = 0.0769585 * v[7]
        v[19] += v[6]
        v[19] += 0.649704;
        v[19] = v[19] - 0.435033*x[0]
        v[19] -= 0.688583*x[1]
        v[19] += 0.222278*x[2]
        v[19] -= 0.524653*x[3]
        v[19] += 0.413248*x[4]

        v[6] = 0.354033 * x[0]
        v[7] = -0.0230349 * x[1]
        v[6] += v[7]
        v[7] = -0.211938 * x[2]
        v[6] += v[7]
        v[7] = -0.0554288 * x[3]
        v[6] += v[7]
        v[7] = 0.220429 * x[4]
        v[6] += v[7]
        v[7] = x[0] * v[6]
        v[6] = 0.1452885 * v[7]
        v[7] = -0.0230349 * x[0]
        v[8] = 0.29135 * x[1]
        v[7] += v[8]
        v[8] = -0.00180333 * x[2]
        v[7] += v[8]
        v[8] = -0.111141 * x[3]
        v[7] += v[8]
        v[8] = 0.0485461 * x[4]
        v[7] += v[8]
        v[8] = x[1] * v[7]
        v[7] = 0.1452885 * v[8]
        v[20] = v[6] + v[7]
        v[7] = -0.211938 * x[0]
        v[8] = -0.00180333 * x[1]
        v[7] += v[8]
        v[8] = 0.815808 * x[2]
        v[7] += v[8]
        v[8] = -0.133538 * x[3]
        v[7] += v[8]
        v[8] = -0.38067 * x[4]
        v[7] += v[8]
        v[8] = x[2] * v[7]
        v[7] = 0.1452885 * v[8]
        v[20] += v[7]
        v[7] = -0.0554288 * x[0]
        v[8] = -0.111141 * x[1]
        v[7] += v[8]
        v[8] = -0.133538 * x[2]
        v[7] += v[8]
        v[8] = 0.389198 * x[3]
        v[7] += v[8]
        v[8] = -0.131586 * x[4]
        v[7] += v[8]
        v[8] = x[3] * v[7]
        v[7] = 0.1452885 * v[8]
        v[20] += v[7]
        v[7] = 0.220429 * x[0]
        v[8] = 0.0485461 * x[1]
        v[7] += v[8]
        v[8] = -0.38067 * x[2]
        v[7] += v[8]
        v[8] = -0.131586 * x[3]
        v[7] += v[8]
        v[8] = 0.534706 * x[4]
        v[7] += v[8]
        v[8] = x[4] * v[7]
        v[7] = 0.1452885 * v[8]
        v[20] += v[7]
        v[20] += 0.344144;
        v[20] = v[20] + 0.759468*x[0]
        v[20] -= 0.627795*x[1]
        v[20] += 0.0403142*x[2]
        v[20] += 0.724666*x[3]
        v[20] -= 0.0182537*x[4]

        v[7] = 0.354033 * x[0]
        v[8] = -0.0230349 * x[1]
        v[7] += v[8]
        v[8] = -0.211938 * x[2]
        v[7] += v[8]
        v[8] = -0.0554288 * x[3]
        v[7] += v[8]
        v[8] = 0.220429 * x[4]
        v[7] += v[8]
        v[8] = x[0] * v[7]
        v[7] = -0.079689 * v[8]
        v[8] = -0.0230349 * x[0]
        v[9] = 0.29135 * x[1]
        v[8] += v[9]
        v[9] = -0.00180333 * x[2]
        v[8] += v[9]
        v[9] = -0.111141 * x[3]
        v[8] += v[9]
        v[9] = 0.0485461 * x[4]
        v[8] += v[9]
        v[9] = x[1] * v[8]
        v[8] = -0.079689 * v[9]
        v[21] = v[7] + v[8]
        v[8] = -0.211938 * x[0]
        v[9] = -0.00180333 * x[1]
        v[8] += v[9]
        v[9] = 0.815808 * x[2]
        v[8] += v[9]
        v[9] = -0.133538 * x[3]
        v[8] += v[9]
        v[9] = -0.38067 * x[4]
        v[8] += v[9]
        v[9] = x[2] * v[8]
        v[8] = -0.079689 * v[9]
        v[21] += v[8]
        v[8] = -0.0554288 * x[0]
        v[9] = -0.111141 * x[1]
        v[8] += v[9]
        v[9] = -0.133538 * x[2]
        v[8] += v[9]
        v[9] = 0.389198 * x[3]
        v[8] += v[9]
        v[9] = -0.131586 * x[4]
        v[8] += v[9]
        v[9] = x[3] * v[8]
        v[8] = -0.079689 * v[9]
        v[21] += v[8]
        v[8] = 0.220429 * x[0]
        v[9] = 0.0485461 * x[1]
        v[8] += v[9]
        v[9] = -0.38067 * x[2]
        v[8] += v[9]
        v[9] = -0.131586 * x[3]
        v[8] += v[9]
        v[9] = 0.534706 * x[4]
        v[8] += v[9]
        v[9] = x[4] * v[8]
        v[8] = -0.079689 * v[9]
        v[21] += v[8]
        v[21] += -0.627443;
        v[21] = v[21] - 0.152448*x[0]
        v[21] -= 0.546437*x[1]
        v[21] += 0.484134*x[2]
        v[21] += 0.353951*x[3]
        v[21] += 0.887866*x[4]

        v[8] = 0.354033 * x[0]
        v[9] = -0.0230349 * x[1]
        v[8] += v[9]
        v[9] = -0.211938 * x[2]
        v[8] += v[9]
        v[9] = -0.0554288 * x[3]
        v[8] += v[9]
        v[9] = 0.220429 * x[4]
        v[8] += v[9]
        v[9] = x[0] * v[8]
        v[8] = 27.3455 * v[9]
        v[9] = -0.0230349 * x[0]
        v[10] = 0.29135 * x[1]
        v[9] += v[10]
        v[10] = -0.00180333 * x[2]
        v[9] += v[10]
        v[10] = -0.111141 * x[3]
        v[9] += v[10]
        v[10] = 0.0485461 * x[4]
        v[9] += v[10]
        v[10] = x[1] * v[9]
        v[9] = 27.3455 * v[10]
        v[22] = v[8] + v[9]
        v[9] = -0.211938 * x[0]
        v[10] = -0.00180333 * x[1]
        v[9] += v[10]
        v[10] = 0.815808 * x[2]
        v[9] += v[10]
        v[10] = -0.133538 * x[3]
        v[9] += v[10]
        v[10] = -0.38067 * x[4]
        v[9] += v[10]
        v[10] = x[2] * v[9]
        v[9] = 27.3455 * v[10]
        v[22] += v[9]
        v[9] = -0.0554288 * x[0]
        v[10] = -0.111141 * x[1]
        v[9] += v[10]
        v[10] = -0.133538 * x[2]
        v[9] += v[10]
        v[10] = 0.389198 * x[3]
        v[9] += v[10]
        v[10] = -0.131586 * x[4]
        v[9] += v[10]
        v[10] = x[3] * v[9]
        v[9] = 27.3455 * v[10]
        v[22] += v[9]
        v[9] = 0.220429 * x[0]
        v[10] = 0.0485461 * x[1]
        v[9] += v[10]
        v[10] = -0.38067 * x[2]
        v[9] += v[10]
        v[10] = -0.131586 * x[3]
        v[9] += v[10]
        v[10] = 0.534706 * x[4]
        v[9] += v[10]
        v[10] = x[4] * v[9]
        v[9] = 27.3455 * v[10]
        v[22] += v[9]
        v[22] += 0.001828;
        v[22] = v[22] - 0.821772*x[0]
        v[22] -= 0.53412*x[1]
        v[22] -= 0.798498*x[2]
        v[22] -= 0.658572*x[3]
        v[22] += 0.662362*x[4]

        v[9] = 0.354033 * x[0]
        v[10] = -0.0230349 * x[1]
        v[9] += v[10]
        v[10] = -0.211938 * x[2]
        v[9] += v[10]
        v[10] = -0.0554288 * x[3]
        v[9] += v[10]
        v[10] = 0.220429 * x[4]
        v[9] += v[10]
        v[10] = x[0] * v[9]
        v[9] = -0.2224365 * v[10]
        v[10] = -0.0230349 * x[0]
        v[11] = 0.29135 * x[1]
        v[10] += v[11]
        v[11] = -0.00180333 * x[2]
        v[10] += v[11]
        v[11] = -0.111141 * x[3]
        v[10] += v[11]
        v[11] = 0.0485461 * x[4]
        v[10] += v[11]
        v[11] = x[1] * v[10]
        v[10] = -0.2224365 * v[11]
        v[23] = v[9] + v[10]
        v[10] = -0.211938 * x[0]
        v[11] = -0.00180333 * x[1]
        v[10] += v[11]
        v[11] = 0.815808 * x[2]
        v[10] += v[11]
        v[11] = -0.133538 * x[3]
        v[10] += v[11]
        v[11] = -0.38067 * x[4]
        v[10] += v[11]
        v[11] = x[2] * v[10]
        v[10] = -0.2224365 * v[11]
        v[23] += v[10]
        v[10] = -0.0554288 * x[0]
        v[11] = -0.111141 * x[1]
        v[10] += v[11]
        v[11] = -0.133538 * x[2]
        v[10] += v[11]
        v[11] = 0.389198 * x[3]
        v[10] += v[11]
        v[11] = -0.131586 * x[4]
        v[10] += v[11]
        v[11] = x[3] * v[10]
        v[10] = -0.2224365 * v[11]
        v[23] += v[10]
        v[10] = 0.220429 * x[0]
        v[11] = 0.0485461 * x[1]
        v[10] += v[11]
        v[11] = -0.38067 * x[2]
        v[10] += v[11]
        v[11] = -0.131586 * x[3]
        v[10] += v[11]
        v[11] = 0.534706 * x[4]
        v[10] += v[11]
        v[11] = x[4] * v[10]
        v[10] = -0.2224365 * v[11]
        v[23] += v[10]
        v[23] += -0.224783;
        v[23] = v[23] + 0.819831*x[0]
        v[23] -= 0.910632*x[1]
        v[23] -= 0.480344*x[2]
        v[23] -= 0.871758*x[3]
        v[23] -= 0.978666*x[4]

        v[12] = v[14] * v[14]
        v[13] = v[15] * v[15]
        v[12] += v[13]
        v[13] = v[16] * v[16]
        v[12] += v[13]
        v[13] = v[17] * v[17]
        v[12] += v[13]
        v[13] = v[18] * v[18]
        v[12] += v[13]
        v[13] = v[19] * v[19]
        v[12] += v[13]
        v[13] = v[20] * v[20]
        v[12] += v[13]
        v[13] = v[21] * v[21]
        v[12] += v[13]
        v[13] = v[22] * v[22]
        v[12] += v[13]
        v[13] = v[23] * v[23]
        v[12] += v[13]
        return v[12]

# needs more precise floats
class s267(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.2232795288, 11.7767204712],
                [6.1236156871, 26.1236156871], [-10.5942083977, 9.4057916023],
                [-5.2928287845, 14.7071712155], [-8.2232796262, 0]]

    def _function(self,x):
        term1 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        y = numpy.sum((term1 + numpy.exp(numpy.multiply(term2,x[0]))*x[2] -
                       numpy.exp(numpy.multiply(term2,x[1]))*x[3] +
                       3*numpy.exp(numpy.multiply(term2,x[4])))**2)
        return y

# needs more precise floats
class s358(BenchmarkFunction):
    @property
    def domain(self):
        return [[-0.5, 0.45], [1.5, 2.25],
                [-2, -0.9], [0.001, 0.09], [0.001, 0.09]]

    def _function(self,x):
        term1 = [0.844,0.908,0.932,0.936,0.925,0.908,0.881,0.85,0.818,0.784,
                 0.751,0.718,0.685,0.658,0.628,0.603,0.58,0.558,0.538,
                 0.522,0.506,0.49,0.478,0.467,0.457,0.448,0.438,0.431,0.424,
                 0.420,0.414,0.411,0.406]
        term2 = [round(x*-10,ndigits=1) for x in range(len(term1))]
        return numpy.sum((term1 - numpy.exp(numpy.multiply(term2,x[3]))*x[1] -
                          numpy.exp(numpy.multiply(term2,x[4]))*x[2] -x[0])**2)

# new
class schwefel(BenchmarkFunction):
    @property
    def domain(self):
        return [[-0.5, 0.36], [-0.5, 0.36], [-0.5, 0.36], [-0.5, 0.36], [-0.5, 0.36]]

    def _function(self,x):
        v = numpy.zeros(2)
        v[0] = pow(x[0], 10)
        v[1] = pow(x[1], 10)
        v[0] += v[1]
        v[1] = pow(x[2], 10)
        v[0] += v[1]
        v[1] = pow(x[3], 10)
        v[0] += v[1]
        v[1] = pow(x[4], 10)
        v[0] += v[1]

        return v[0]

# needs more precise floats
class biggs6(BenchmarkFunction):
    @property
    def domain(self):
        return [[-8.2885839998, 10.54027440018],
                [7.6831983277, 24.91487849493],
                [-3.05828543, 15.247543113], [-4.8134383873, 13.66790545143],
                [-8.2885839966, 10.54027440306],
                [-14.6154272503, 4.84611547473]]

    def _function(self,x):
        term1 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334,-0.325852731997495,
                 -0.28907018464926]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        return numpy.sum((term1 + numpy.exp(numpy.multiply(term2,x[0]))*x[2] -
                          numpy.exp(numpy.multiply(term2,x[1]))*x[3] +
                          numpy.exp(numpy.multiply(term2,x[4]))*x[5])**2)

class hart6(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        
    def _function(self,x):
        coeff = numpy.array([[10,0.05,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],
                 [3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
        term = numpy.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
                            [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
                            [0.2348,0.1451,0.3522,0.2883,0.3047,0.665],
                            [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
        coeff_exp = numpy.array([-1,-1.2,-3,-3.2])
        y = numpy.sum(coeff_exp*numpy.exp(-(numpy.sum(coeff*(x-term)**2,
                                                      axis=1))))
        return y

# new
class heart6ls(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.9969001405, 9.00278987355], [-10.0002239415, 8.99979845265], [-7.3184801716, 11.41336784556], [-7.7497837397, 11.02519463427], [-30.2417255436, -9.21755298924], [-9.2029016565, 9.71738850915]]

    def _function(self,x):
        v = numpy.zeros(6)
        v[0] = x[2] * x[0]
        v[1] = -0.816 - x[0]
        v[2] = x[3] * v[1]
        v[1] = v[0] + v[2]
        v[0] = x[4] * x[1]
        v[2] = v[1] - v[0]
        v[1] = -0.017 - x[1]
        v[0] = x[5] * v[1]
        v[1] = v[2] - v[0]
        v[2] = 1.826 + v[1]
        v[1] = v[2] * v[2]
        v[2] = x[4] * x[0]
        v[2] += 0.754;
        v[0] = -0.816 - x[0]
        v[3] = x[5] * v[0]
        v[2] += v[3]
        v[3] = x[2] * x[1]
        v[2] += v[3]
        v[3] = -0.017 - x[1]
        v[0] = x[3] * v[3]
        v[2] += v[0]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = x[2] * x[2]
        v[2] = x[4] * x[4]
        v[3] = v[0] - v[2]
        v[0] = x[0] * v[3]
        v[3] = 2. * x[1]
        v[2] = v[3] * x[2]
        v[3] = v[2] * x[4]
        v[2] = v[0] - v[3]
        v[0] = -0.816 - x[0]
        v[3] = x[3] * x[3]
        v[4] = x[5] * x[5]
        v[5] = v[3] - v[4]
        v[3] = v[0] * v[5]
        v[0] = v[2] + v[3]
        v[2] = -0.017 - x[1]
        v[3] = 2. * v[2]
        v[2] = v[3] * x[3]
        v[3] = v[2] * x[5]
        v[2] = v[0] - v[3]
        v[0] = 4.839 + v[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[4] * x[4]
        v[3] = v[2] - v[0]
        v[2] = x[1] * v[3]
        v[2] += 3.259;
        v[3] = 2. * x[0]
        v[0] = v[3] * x[2]
        v[3] = v[0] * x[4]
        v[2] += v[3]
        v[3] = -0.017 - x[1]
        v[0] = x[3] * x[3]
        v[5] = x[5] * x[5]
        v[4] = v[0] - v[5]
        v[0] = v[3] * v[4]
        v[2] += v[0]
        v[0] = -0.816 - x[0]
        v[3] = 2. * v[0]
        v[0] = v[3] * x[3]
        v[3] = v[0] * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[1] += v[3]
        v[3] = x[0] * x[2]
        v[2] = x[2] * x[2]
        v[0] = x[4] * x[4]
        v[4] = -3. * v[0]
        v[0] = v[2] + v[4]
        v[2] = v[3] * v[0]
        v[2] += 14.023;
        v[3] = x[1] * x[4]
        v[0] = x[4] * x[4]
        v[4] = x[2] * x[2]
        v[5] = -3. * v[4]
        v[4] = v[0] + v[5]
        v[0] = v[3] * v[4]
        v[2] += v[0]
        v[0] = -0.816 - x[0]
        v[3] = v[0] * x[3]
        v[0] = x[3] * x[3]
        v[4] = x[5] * x[5]
        v[5] = -3. * v[4]
        v[4] = v[0] + v[5]
        v[0] = v[3] * v[4]
        v[2] += v[0]
        v[0] = -0.017 - x[1]
        v[3] = v[0] * x[5]
        v[0] = x[5] * x[5]
        v[4] = x[3] * x[3]
        v[5] = -3. * v[4]
        v[4] = v[0] + v[5]
        v[0] = v[3] * v[4]
        v[2] += v[0]
        v[0] = v[2] * v[2]
        v[1] += v[0]
        v[0] = x[1] * x[2]
        v[2] = x[2] * x[2]
        v[3] = x[4] * x[4]
        v[4] = -3. * v[3]
        v[3] = v[2] + v[4]
        v[2] = v[0] * v[3]
        v[0] = x[0] * x[4]
        v[3] = x[4] * x[4]
        v[4] = x[2] * x[2]
        v[5] = -3. * v[4]
        v[4] = v[3] + v[5]
        v[3] = v[0] * v[4]
        v[0] = v[2] - v[3]
        v[2] = -0.017 - x[1]
        v[3] = v[2] * x[3]
        v[2] = x[3] * x[3]
        v[4] = x[5] * x[5]
        v[5] = -3. * v[4]
        v[4] = v[2] + v[5]
        v[2] = v[3] * v[4]
        v[3] = v[0] + v[2]
        v[0] = -0.816 - x[0]
        v[2] = v[0] * x[5]
        v[0] = x[5] * x[5]
        v[4] = x[3] * x[3]
        v[5] = -3. * v[4]
        v[4] = v[0] + v[5]
        v[0] = v[2] * v[4]
        v[2] = v[3] - v[0]
        v[3] = -15.467 + v[2]
        v[2] = v[3] * v[3]
        v[1] += v[2]
        return v[1]

class palmer2a(BenchmarkFunction):
    @property
    def domain(self):
        return [[0,32.4286981517],[1e-10,10.7435278989],
                [-20.7797273226,-0.779727322599999],
                [-25.3729423234,-5.3729423234],
                [3.6520539577,23.6520539577],
                [-10.081937131,9.918062869]]

    def _function(self,x):
        term0 = [72.676767,40.149455,18.8548,6.4762,0.8596,0,0.273,
                 3.2043,8.108,13.4291,17.714,19.4529,17.7149,13.4291,
                 8.108,3.2053,0.273,0,0.8596,6.4762,18.8548,40.149455,
                 72.676767]
        term1_3 = [3.046173318241,2.467400073616,1.949550365169,1.4926241929,
                   1.096623651204,0.878319472969,0.761544202225,0.487388289424,
                   0.274155912801,0.121847072356,0.030461768089,0,
                   0.030461768089,0.121847072356,0.274155912801,0.487388289424,
                   0.761544202225,0.878319472969,1.096623651204,1.4926241929,
                   1.949550365169,2.467400073616,3.046173318241]
        term4 = [9.27917188476338,6.08806312328024,3.80074662633058,
                 2.22792698123038,1.20258343237999,0.771445096596542,
                 0.579949571942512,0.237547344667653,0.0751614645237495,
                 0.0148467090417283,0.000927919315108019,0,
                 0.000927919315108019,0.0148467090417283,0.0751614645237495,
                 0.237547344667653,0.579949571942512,0.771445096596542,
                 1.20258343237999,2.22792698123038,3.80074662633058,
                 6.08806312328024,9.27917188476338]
        term5 = [28.2659658107383,15.0216873985605,7.40974697327763,
                 3.32545771219912,1.31878143449399,0.677575250667194,
                 0.44165723409569,0.115777793974781,0.0206059599139685,
                 0.00180902803085595,2.82660629821242e-5,0,
                 2.82660629821242e-5,0.00180902803085595,0.0206059599139685,
                 0.115777793974781,0.44165723409569,0.677575250667194,
                 1.31878143449399,3.32545771219912,7.40974697327763,
                 15.0216873985605,28.2659658107383]
        y = numpy.sum((term0 - x[0]/(term1_3 + x[1]) - x[2] -
                       numpy.multiply(term1_3,x[3]) -
                       numpy.multiply(term4,x[4]) -
                       numpy.multiply(term5,x[5]))**2)
        return y

# needs more precise floats
class palmer5c(BenchmarkFunction):
    @property
    def domain(self):
        return [[27.5370157298, 42.78331415682],
                [-11.7302338172, 7.44278956452],
                [30.7938174564, 45.71443571076],
                [-9.1697871977, 9.74719152207],
                [-6.2910484684, 12.33805637844],
                [-10.1772297675, 8.84049320925]]

    def _function(self,x):
        term0 = [83.57418,81.007654,18.983286,8.051067,2.044762,0,1.170451,
                 10.479881,25.785001,44.126844,62.822177,77.719674]
        term1 = [ 1,                 -1,                  -0.580246662076097,
                 -0.388889596244429, -0.209876103077636,  -0.0274008344074166,
                  0.111109979342031,  0.382715263431966,   0.604937768596458,
                  0.777777494835508,  0.901234442149115,   0.975308610537279]
        term2 = [-1                , -1               ,    0.326627622299095,
                  0.697529763865691,  0.911904042713891,   0.998498388547555,
                  0.975309144981227,  0.707058054272402,   0.268100592251076,
                 -0.209875662945196, -0.624447039431652,  -0.902453771576314]
        term3 = [ 1,                 -1.00000000000001 ,   0.959295837237901,
                  0.931413732720829,  0.592649836808703,   0.0821202124087442,
                 -0.327843137243947, -0.923919082437077,  -0.929306116667968,
                 -0.451304360130595,  0.224311916318587,   0.785033257523164]
        term4 = [-1,                 -1.00000000000001,    0.786628792702478,
                  0.0269044570429476,-0.663137966235876,  -0.993998063864127,
                 -0.902455856568024,  1.37815777051409e-4, 0.856244144869245,
                  0.911904412206629,  0.22013178989009,   -0.628845619664628]
        term5 = [ 1,                 -1.00000000000002,   -0.0464183747207751,
                 -0.910488005847614, -0.871003461121532,  -0.136592965107211,
                  0.528386840404683,  0.923813594034239,  -0.106642728074,
                 -0.96721309838044,  -0.62109261804035,    0.441603837591961]
        y = numpy.sum((term0 - x[0] + numpy.multiply(term1,x[1]) +
                       numpy.multiply(term2,x[2])+numpy.multiply(term3,x[3]) +
                       numpy.multiply(term4,x[4]) +
                       numpy.multiply(term5,x[5]))**2)
        return y

# needs more precise floats
class palmer6a(BenchmarkFunction):
    @property
    def domain(self):
        return [[-44.1581372624, -24.1581372624], [0.120997701, 20.120997701],
                [-1.1808208888, 18.8191791112],
                [-8.633061426, 11.366938574], [1e-05, 43.2710391882],
                [1e-05, 10.7437425261]]

    def _function(self,x):
        term0 = [10.678659,75.414511,41.513459,20.104735,7.432436,1.298082,
                 0.1713,0,0.068203,0.774499,2.070002,5.574556,9.026378]
        term1_5 = [0,2.467400073616,1.949550365169,1.4926241929,1.096623651204,
                   0.761544202225,0.616850018404,0.536979718521,0.487388289424,
                   0.373156048225,0.274155912801,0.121847072356,0.030461768089]
        term2 = [0,6.08806312328024,3.80074662633058,2.22792698123038,
                 1.20258343237999,0.579949571942512,0.380503945205015,
                 0.288347218102892,0.237547344667653,0.139245436326899,
                 0.0751614645237495,0.0148467090417283,9.27919315108019e-4]
        term3 = [0,15.0216873985605,7.40974697327763,3.32545771219912,
                 1.31878143449399,0.44165723409569,0.234713865602508,
                 0.154836608013205,0.115777793974781,0.0519602767531113,
                 0.0206059599139685,0.00180902803085595,2.82660629821242e-5]
        y = numpy.sum((term0 - x[4]/(term1_5 + x[5]) - x[0] -
                       numpy.multiply(term1_5,x[1]) -numpy.multiply(term2,x[2]) -
                       numpy.multiply(term3,x[3]))**2)
        return y

# needs more precise floats
class palmer8a(BenchmarkFunction):
    @property
    def domain(self):
        return [[1e-05, 12.4961104793], [1e-05, 10.2011908033],
                [-17.7129671187, 2.2870328813],
                [-5.0299734848, 14.9700265152],
                [2.8287670723, 22.8287670723], [-9.0495003432, 10.9504996568]]

    def _function(self,x):
        term0 = [4.757534,3.121416,1.207606,0.131916,0,0.258514,
                 3.380161,10.762813,23.745996,44.471864,76.541947,97.874528]
        term1_3 = [0,0.030461768089,0.098695877281,0.190385614224,
                   0.264714366016,0.373156048225,0.616850018404,0.921467524761,
                   1.287008567296,1.713473146009,2.2008612609,2.467400073616]
        term4 = [0,9.27919315108019e-4,0.00974087619226621,0.0362466821034498,
                 0.0700736955752528,0.139245436326899,0.380503945205015,
                 0.849102399189164,1.6563910522933,2.93599022209398,
                 4.84379028973034,6.08806312328024]
        term5 = [0,2.82660629821242e-5,9.61384321281321e-4,0.00690084683584735,
                 0.0185495138986012,0.0519602767531113,0.234713865602508,
                 0.782420286049465,2.13178947509392,5.03074040250303,
                 10.6605104045911,15.0216873985605]
        y = numpy.sum((term0 - x[0]/(term1_3 + x[1]) - x[2] -
                       numpy.multiply(term1_3,x[3]) -
                       numpy.multiply(term4,x[4]) -
                       numpy.multiply(term5,x[5]))**2)
        return y

# new
class s271(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = -1. + x[0]
        v[1] = v[0] * v[0]
        v[0] = 150. * v[1]
        v[1] = -1. + x[1]
        v[2] = v[1] * v[1]
        v[1] = 140. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[2]
        v[2] = v[1] * v[1]
        v[1] = 130. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[3]
        v[2] = v[1] * v[1]
        v[1] = 120. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[4]
        v[2] = v[1] * v[1]
        v[1] = 110. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[5]
        v[2] = v[1] * v[1]
        v[1] = 100. * v[2]
        v[0] += v[1]
        return v[0]
    
# needs more precise floats
class s272(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 10.999999907], [0, 20.0000005284], [0, 13.9999995189],
                [0, 10.9999998493], [0, 14.9999995532], [0, 12.99999966]]

    def _function(self,x):
        term1 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334,-0.325852731997496,
                 -0.28907018464926]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        return numpy.sum((term1 + numpy.exp(numpy.multiply(term2,x[0]))*x[3] -
                          numpy.exp(numpy.multiply(term2,x[1]))*x[4] +
                          numpy.exp(numpy.multiply(term2,x[2]))*x[5])**2)

# new
class s273(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9], [-9.0, 9.9]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = -1. + x[0]
        v[1] = v[0] * v[0]
        v[0] = 150. * v[1]
        v[1] = -1. + x[1]
        v[2] = v[1] * v[1]
        v[1] = 140. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[2]
        v[2] = v[1] * v[1]
        v[1] = 130. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[3]
        v[2] = v[1] * v[1]
        v[1] = 120. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[4]
        v[2] = v[1] * v[1]
        v[1] = 110. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[5]
        v[2] = v[1] * v[1]
        v[1] = 100. * v[2]
        v[0] += v[1]
        v[1] = -1. + x[0]
        v[2] = v[1] * v[1]
        v[1] = 15. * v[2]
        v[2] = -1. + x[1]
        v[3] = v[2] * v[2]
        v[2] = 14. * v[3]
        v[1] += v[2]
        v[2] = -1. + x[2]
        v[3] = v[2] * v[2]
        v[2] = 13. * v[3]
        v[1] += v[2]
        v[2] = -1. + x[3]
        v[3] = v[2] * v[2]
        v[2] = 12. * v[3]
        v[1] += v[2]
        v[2] = -1. + x[4]
        v[3] = v[2] * v[2]
        v[2] = 11. * v[3]
        v[1] += v[2]
        v[2] = -1. + x[5]
        v[3] = v[2] * v[2]
        v[2] = 10. * v[3]
        v[1] += v[2]
        v[2] = v[1] * v[1]
        v[1] = 10. * v[2]
        v[0] += v[1]
        return v[0]
# new
class s276(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0013714286, 9.9986285714], [-9.96, 10.04], [-10.2742857143, 9.7257142857], [-9.28, 10.72], [-10.8, 0.0], [-9.6832, 10.3168]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = 0.5 * x[1]
        v[0] += x[0]
        v[1] = 0.3333333333333333 * x[2]
        v[0] += v[1]
        v[1] = 0.25 * x[3]
        v[0] += v[1]
        v[1] = 0.2 * x[4]
        v[0] += v[1]
        v[1] = 0.16666666666666666 * x[5]
        v[0] += v[1]
        v[1] = x[0] * v[0]
        v[0] = 0.5 * x[0]
        v[2] = 0.3333333333333333 * x[1]
        v[0] += v[2]
        v[2] = 0.25 * x[2]
        v[0] += v[2]
        v[2] = 0.2 * x[3]
        v[0] += v[2]
        v[2] = 0.16666666666666666 * x[4]
        v[0] += v[2]
        v[2] = 0.14285714285714285 * x[5]
        v[0] += v[2]
        v[2] = x[1] * v[0]
        v[1] += v[2]
        v[2] = 0.3333333333333333 * x[0]
        v[0] = 0.25 * x[1]
        v[2] += v[0]
        v[0] = 0.2 * x[2]
        v[2] += v[0]
        v[0] = 0.16666666666666666 * x[3]
        v[2] += v[0]
        v[0] = 0.14285714285714285 * x[4]
        v[2] += v[0]
        v[0] = 0.125 * x[5]
        v[2] += v[0]
        v[0] = x[2] * v[2]
        v[1] += v[0]
        v[0] = 0.25 * x[0]
        v[2] = 0.2 * x[1]
        v[0] += v[2]
        v[2] = 0.16666666666666666 * x[2]
        v[0] += v[2]
        v[2] = 0.14285714285714285 * x[3]
        v[0] += v[2]
        v[2] = 0.125 * x[4]
        v[0] += v[2]
        v[2] = 0.1111111111111111 * x[5]
        v[0] += v[2]
        v[2] = x[3] * v[0]
        v[1] += v[2]
        v[2] = 0.2 * x[0]
        v[0] = 0.16666666666666666 * x[1]
        v[2] += v[0]
        v[0] = 0.14285714285714285 * x[2]
        v[2] += v[0]
        v[0] = 0.125 * x[3]
        v[2] += v[0]
        v[0] = 0.1111111111111111 * x[4]
        v[2] += v[0]
        v[0] = 0.1 * x[5]
        v[2] += v[0]
        v[0] = x[4] * v[2]
        v[1] += v[0]
        v[0] = 0.16666666666666666 * x[0]
        v[2] = 0.14285714285714285 * x[1]
        v[0] += v[2]
        v[2] = 0.125 * x[2]
        v[0] += v[2]
        v[2] = 0.1111111111111111 * x[3]
        v[0] += v[2]
        v[2] = 0.1 * x[4]
        v[0] += v[2]
        v[2] = 0.09090909090909091 * x[5]
        v[0] += v[2]
        v[2] = x[5] * v[0]
        v[1] += v[2]
        return v[1]
# new
class s294(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.9865749796, 0.0], [-9.0166017712, 10.9833982288], [-9.0278933299, 10.9721066701], [-9.0525625632, 10.9474374368], [-9.1013488151, 10.8986511849], [-9.192426048, 10.807573952]]

    def _function(self,x):
        v = numpy.zeros(3)
        v[0] = x[0] * x[0]
        v[1] = x[1] - v[0]
        v[0] = v[1] * v[1]
        v[1] = 100. * v[0]
        v[0] = 1. - x[0]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[1] * x[1]
        v[0] = x[2] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 100. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[1]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[2] * x[2]
        v[0] = x[3] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 100. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[2]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[3] * x[3]
        v[0] = x[4] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 100. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[3]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        v[2] = x[4] * x[4]
        v[0] = x[5] - v[2]
        v[2] = v[0] * v[0]
        v[0] = 100. * v[2]
        v[1] += v[0]
        v[0] = 1. - x[4]
        v[2] = v[0] * v[0]
        v[1] += v[2]
        return v[1]
# new
class s370(BenchmarkFunction):
    @property
    def domain(self):
        return [[-10.0157250871, 8.98584742161], [-8.9875651305, 9.91119138255], [-10.2329916267, 8.79030753597], [-8.7395699127, 10.13438707857], [-11.5137289215, 7.63764397065], [-9.0070035692, 9.89369678772]]

    def _function(self,x):
        v = numpy.zeros(4)
        v[0] = x[0] * x[0]
        v[1] = x[0] * x[0]
        v[2] = x[1] - v[1]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.06896551724137931 * x[2]
        v[2] += x[1]
        v[1] = 0.00356718192627824 * x[3]
        v[2] += v[1]
        v[1] = 0.00016400836442658574 * x[4]
        v[2] += v[1]
        v[1] = 7.069326052870074e-06 * x[5]
        v[2] += v[1]
        v[1] = 0.034482758620689655 * x[1]
        v[1] += x[0]
        v[3] = 0.0011890606420927466 * x[2]
        v[1] += v[3]
        v[3] = 4.1002091106646436e-05 * x[3]
        v[1] += v[3]
        v[3] = 1.4138652105740149e-06 * x[4]
        v[1] += v[3]
        v[3] = 4.8753972778414304e-08 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.13793103448275862 * x[2]
        v[1] += x[1]
        v[2] = 0.01426872770511296 * x[3]
        v[1] += v[2]
        v[2] = 0.001312066915412686 * x[4]
        v[1] += v[2]
        v[2] = 0.00011310921684592119 * x[5]
        v[1] += v[2]
        v[2] = 0.06896551724137931 * x[1]
        v[2] += x[0]
        v[3] = 0.0047562425683709865 * x[2]
        v[2] += v[3]
        v[3] = 0.0003280167288531715 * x[3]
        v[2] += v[3]
        v[3] = 2.2621843369184238e-05 * x[4]
        v[2] += v[3]
        v[3] = 1.5601271289092577e-06 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.20689655172413793 * x[2]
        v[2] += x[1]
        v[1] = 0.03210463733650416 * x[3]
        v[2] += v[1]
        v[1] = 0.0044282258395178156 * x[4]
        v[2] += v[1]
        v[1] = 0.0005726154102824761 * x[5]
        v[2] += v[1]
        v[1] = 0.10344827586206896 * x[1]
        v[1] += x[0]
        v[3] = 0.01070154577883472 * x[2]
        v[1] += v[3]
        v[3] = 0.0011070564598794539 * x[3]
        v[1] += v[3]
        v[3] = 0.00011452308205649523 * x[4]
        v[1] += v[3]
        v[3] = 1.1847215385154678e-05 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.27586206896551724 * x[2]
        v[1] += x[1]
        v[2] = 0.05707491082045184 * x[3]
        v[1] += v[2]
        v[2] = 0.010496535323301488 * x[4]
        v[1] += v[2]
        v[2] = 0.001809747469534739 * x[5]
        v[1] += v[2]
        v[2] = 0.13793103448275862 * x[1]
        v[2] += x[0]
        v[3] = 0.019024970273483946 * x[2]
        v[2] += v[3]
        v[3] = 0.002624133830825372 * x[3]
        v[2] += v[3]
        v[3] = 0.0003619494939069478 * x[4]
        v[2] += v[3]
        v[3] = 4.992406812509625e-05 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.3448275862068966 * x[2]
        v[2] += x[1]
        v[1] = 0.08917954815695602 * x[3]
        v[2] += v[1]
        v[1] = 0.020501045553323223 * x[4]
        v[2] += v[1]
        v[1] = 0.004418328783043798 * x[5]
        v[2] += v[1]
        v[1] = 0.1724137931034483 * x[1]
        v[1] += x[0]
        v[3] = 0.02972651605231867 * x[2]
        v[1] += v[3]
        v[3] = 0.005125261388330806 * x[3]
        v[1] += v[3]
        v[3] = 0.0008836657566087596 * x[4]
        v[1] += v[3]
        v[3] = 0.00015235616493254478 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.41379310344827586 * x[2]
        v[1] += x[1]
        v[2] = 0.12841854934601665 * x[3]
        v[1] += v[2]
        v[2] = 0.035425806716142524 * x[4]
        v[1] += v[2]
        v[2] = 0.009161846564519618 * x[5]
        v[1] += v[2]
        v[2] = 0.20689655172413793 * x[1]
        v[2] += x[0]
        v[3] = 0.04280618311533888 * x[2]
        v[2] += v[3]
        v[3] = 0.008856451679035631 * x[3]
        v[2] += v[3]
        v[3] = 0.0018323693129039236 * x[4]
        v[2] += v[3]
        v[3] = 0.0003791108923249497 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.4827586206896552 * x[2]
        v[2] += x[1]
        v[1] = 0.1747919143876338 * x[3]
        v[2] += v[1]
        v[1] = 0.05625486899831893 * x[4]
        v[2] += v[1]
        v[1] = 0.016973451852941055 * x[5]
        v[2] += v[1]
        v[1] = 0.2413793103448276 * x[1]
        v[1] += x[0]
        v[3] = 0.0582639714625446 * x[2]
        v[1] += v[3]
        v[3] = 0.014063717249579732 * x[3]
        v[1] += v[3]
        v[3] = 0.0033946903705882113 * x[4]
        v[1] += v[3]
        v[3] = 0.0008194080204868096 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.5517241379310345 * x[2]
        v[1] += x[1]
        v[2] = 0.22829964328180735 * x[3]
        v[1] += v[2]
        v[2] = 0.0839722825864119 * x[4]
        v[1] += v[2]
        v[2] = 0.028955959512555824 * x[5]
        v[1] += v[2]
        v[2] = 0.27586206896551724 * x[1]
        v[2] += x[0]
        v[3] = 0.07609988109393578 * x[2]
        v[2] += v[3]
        v[3] = 0.020993070646602975 * x[3]
        v[2] += v[3]
        v[3] = 0.005791191902511165 * x[4]
        v[2] += v[3]
        v[3] = 0.00159757018000308 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.6206896551724138 * x[2]
        v[2] += x[1]
        v[1] = 0.28894173602853745 * x[3]
        v[2] += v[1]
        v[1] = 0.11956209766698103 * x[4]
        v[2] += v[1]
        v[1] = 0.046381848232880565 * x[5]
        v[2] += v[1]
        v[1] = 0.3103448275862069 * x[1]
        v[1] += x[0]
        v[3] = 0.09631391200951249 * x[2]
        v[1] += v[3]
        v[3] = 0.029890524416745258 * x[3]
        v[1] += v[3]
        v[3] = 0.009276369646576113 * x[4]
        v[1] += v[3]
        v[3] = 0.002878873338592587 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.6896551724137931 * x[2]
        v[1] += x[1]
        v[2] = 0.35671819262782406 * x[3]
        v[1] += v[2]
        v[2] = 0.16400836442658578 * x[4]
        v[1] += v[2]
        v[2] = 0.07069326052870077 * x[5]
        v[1] += v[2]
        v[2] = 0.3448275862068966 * x[1]
        v[2] += x[0]
        v[3] = 0.11890606420927469 * x[2]
        v[2] += v[3]
        v[3] = 0.041002091106646446 * x[3]
        v[2] += v[3]
        v[3] = 0.014138652105740154 * x[4]
        v[2] += v[3]
        v[3] = 0.004875397277841433 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.7586206896551724 * x[2]
        v[2] += x[1]
        v[1] = 0.431629013079667 * x[3]
        v[2] += v[1]
        v[1] = 0.2182951330517856 * x[4]
        v[2] += v[1]
        v[1] = 0.10350200274007074 * x[5]
        v[2] += v[1]
        v[1] = 0.3793103448275862 * x[1]
        v[1] += x[0]
        v[3] = 0.14387633769322233 * x[2]
        v[1] += v[3]
        v[3] = 0.0545737832629464 * x[3]
        v[1] += v[3]
        v[3] = 0.02070040054801415 * x[4]
        v[1] += v[3]
        v[3] = 0.007851876069936401 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.8275862068965517 * x[2]
        v[1] += x[1]
        v[2] = 0.5136741973840666 * x[3]
        v[1] += v[2]
        v[2] = 0.2834064537291402 * x[4]
        v[1] += v[2]
        v[2] = 0.1465895450323139 * x[5]
        v[1] += v[2]
        v[2] = 0.41379310344827586 * x[1]
        v[2] += x[0]
        v[3] = 0.17122473246135553 * x[2]
        v[2] += v[3]
        v[3] = 0.07085161343228505 * x[3]
        v[2] += v[3]
        v[3] = 0.029317909006462778 * x[4]
        v[2] += v[3]
        v[3] = 0.01213154855439839 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 0.896551724137931 * x[2]
        v[2] += x[1]
        v[1] = 0.6028537455410227 * x[3]
        v[2] += v[1]
        v[1] = 0.36032637664520895 * x[4]
        v[2] += v[1]
        v[1] = 0.20190702139602226 * x[5]
        v[2] += v[1]
        v[1] = 0.4482758620689655 * x[1]
        v[1] += x[0]
        v[3] = 0.2009512485136742 * x[2]
        v[1] += v[3]
        v[3] = 0.09008159416130224 * x[3]
        v[1] += v[3]
        v[3] = 0.040381404279204454 * x[4]
        v[1] += v[3]
        v[3] = 0.01810200881481579 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 0.9655172413793104 * x[2]
        v[1] += x[1]
        v[2] = 0.6991676575505352 * x[3]
        v[1] += v[2]
        v[2] = 0.4500389519865514 * x[4]
        v[1] += v[2]
        v[2] = 0.2715752296470569 * x[5]
        v[1] += v[2]
        v[2] = 0.4827586206896552 * x[1]
        v[2] += x[0]
        v[3] = 0.2330558858501784 * x[2]
        v[2] += v[3]
        v[3] = 0.11250973799663785 * x[3]
        v[2] += v[3]
        v[3] = 0.05431504592941138 * x[4]
        v[2] += v[3]
        v[3] = 0.026221056655577907 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.0344827586206897 * x[2]
        v[2] += x[1]
        v[1] = 0.8026159334126042 * x[3]
        v[2] += v[1]
        v[1] = 0.5535282299397271 * x[4]
        v[2] += v[1]
        v[1] = 0.3578846314265477 * x[5]
        v[2] += v[1]
        v[1] = 0.5172413793103449 * x[1]
        v[1] += x[0]
        v[3] = 0.26753864447086806 * x[2]
        v[1] += v[3]
        v[3] = 0.13838205748493176 * x[3]
        v[1] += v[3]
        v[3] = 0.07157692628530954 * x[4]
        v[1] += v[3]
        v[3] = 0.037022548078608386 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.103448275862069 * x[2]
        v[1] += x[1]
        v[2] = 0.9131985731272294 * x[3]
        v[1] += v[2]
        v[2] = 0.6717782606912952 * x[4]
        v[1] += v[2]
        v[2] = 0.4632953522008932 * x[5]
        v[1] += v[2]
        v[2] = 0.5517241379310345 * x[1]
        v[2] += x[0]
        v[3] = 0.30439952437574314 * x[2]
        v[2] += v[3]
        v[3] = 0.1679445651728238 * x[3]
        v[2] += v[3]
        v[3] = 0.09265907044017864 * x[4]
        v[2] += v[3]
        v[3] = 0.05112224576009856 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.1724137931034482 * x[2]
        v[2] += x[1]
        v[1] = 1.0309155766944111 * x[3]
        v[2] += v[1]
        v[1] = 0.8057730944278155 * x[4]
        v[2] += v[1]
        v[1] = 0.5904371812617614 * x[5]
        v[2] += v[1]
        v[1] = 0.5862068965517241 * x[1]
        v[1] += x[0]
        v[3] = 0.34363852556480373 * x[2]
        v[1] += v[3]
        v[3] = 0.20144327360695388 * x[3]
        v[1] += v[3]
        v[3] = 0.11808743625235227 * x[4]
        v[1] += v[3]
        v[3] = 0.06922366952724097 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.2413793103448276 * x[2]
        v[1] += x[1]
        v[2] = 1.1557669441141498 * x[3]
        v[1] += v[2]
        v[2] = 0.9564967813358483 * x[4]
        v[1] += v[2]
        v[2] = 0.742109571726089 * x[5]
        v[1] += v[2]
        v[2] = 0.6206896551724138 * x[1]
        v[2] += x[0]
        v[3] = 0.38525564803804996 * x[2]
        v[2] += v[3]
        v[3] = 0.23912419533396206 * x[3]
        v[2] += v[3]
        v[3] = 0.14842191434521781 * x[4]
        v[2] += v[3]
        v[3] = 0.09212394683496278 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.3103448275862069 * x[2]
        v[2] += x[1]
        v[1] = 1.2877526753864446 * x[3]
        v[2] += v[1]
        v[1] = 1.1249333716019516 * x[4]
        v[2] += v[1]
        v[1] = 0.9212816405360811 * x[5]
        v[2] += v[1]
        v[1] = 0.6551724137931034 * x[1]
        v[1] += x[0]
        v[3] = 0.42925089179548154 * x[2]
        v[1] += v[3]
        v[3] = 0.2812333429004879 * x[3]
        v[1] += v[3]
        v[3] = 0.1842563281072162 * x[4]
        v[1] += v[3]
        v[3] = 0.12071966324265888 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.3793103448275863 * x[2]
        v[1] += x[1]
        v[2] = 1.4268727705112962 * x[3]
        v[1] += v[2]
        v[2] = 1.3120669154126863 * x[4]
        v[1] += v[2]
        v[2] = 1.1310921684592123 * x[5]
        v[1] += v[2]
        v[2] = 0.6896551724137931 * x[1]
        v[2] += x[0]
        v[3] = 0.47562425683709875 * x[2]
        v[2] += v[3]
        v[3] = 0.32801672885317157 * x[3]
        v[2] += v[3]
        v[3] = 0.22621843369184247 * x[4]
        v[2] += v[3]
        v[3] = 0.15601271289092586 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.4482758620689655 * x[2]
        v[2] += x[1]
        v[1] = 1.573127229488704 * x[3]
        v[2] += v[1]
        v[1] = 1.5188814629546108 * x[4]
        v[2] += v[1]
        v[1] = 1.3748496000882255 * x[5]
        v[2] += v[1]
        v[1] = 0.7241379310344828 * x[1]
        v[1] += x[0]
        v[3] = 0.5243757431629014 * x[2]
        v[1] += v[3]
        v[3] = 0.3797203657386527 * x[3]
        v[1] += v[3]
        v[3] = 0.2749699200176451 * x[4]
        v[1] += v[3]
        v[3] = 0.19911614897829472 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.5172413793103448 * x[2]
        v[1] += x[1]
        v[2] = 1.726516052318668 * x[3]
        v[1] += v[2]
        v[2] = 1.7463610644142848 * x[4]
        v[1] += v[2]
        v[2] = 1.6560320438411318 * x[5]
        v[1] += v[2]
        v[2] = 0.7586206896551724 * x[1]
        v[2] += x[0]
        v[3] = 0.5755053507728893 * x[2]
        v[2] += v[3]
        v[3] = 0.4365902661035712 * x[3]
        v[2] += v[3]
        v[3] = 0.3312064087682264 * x[4]
        v[2] += v[3]
        v[3] = 0.25126003423796484 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.5862068965517242 * x[2]
        v[2] += x[1]
        v[1] = 1.8870392390011892 * x[3]
        v[2] += v[1]
        v[1] = 1.9954897699782692 * x[4]
        v[2] += v[1]
        v[1] = 1.9782872719612155 * x[5]
        v[2] += v[1]
        v[1] = 0.7931034482758621 * x[1]
        v[1] += x[0]
        v[3] = 0.6290130796670631 * x[2]
        v[1] += v[3]
        v[3] = 0.4988724424945673 * x[3]
        v[1] += v[3]
        v[3] = 0.3956574543922431 * x[4]
        v[1] += v[3]
        v[3] = 0.31379729141453766 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.6551724137931034 * x[2]
        v[1] += x[1]
        v[2] = 2.0546967895362664 * x[3]
        v[1] += v[2]
        v[2] = 2.2672516298331216 * x[4]
        v[1] += v[2]
        v[2] = 2.3454327205170222 * x[5]
        v[1] += v[2]
        v[2] = 0.8275862068965517 * x[1]
        v[2] += x[0]
        v[3] = 0.6848989298454221 * x[2]
        v[2] += v[3]
        v[3] = 0.5668129074582804 * x[3]
        v[2] += v[3]
        v[3] = 0.46908654410340445 * x[4]
        v[2] += v[3]
        v[3] = 0.3882095537407485 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.7241379310344827 * x[2]
        v[2] += x[1]
        v[1] = 2.2294887039238995 * x[3]
        v[2] += v[1]
        v[1] = 2.562630694165402 * x[4]
        v[2] += v[1]
        v[1] = 2.7614554894023726 * x[5]
        v[2] += v[1]
        v[1] = 0.8620689655172413 * x[1]
        v[1] += x[0]
        v[3] = 0.7431629013079666 * x[2]
        v[1] += v[3]
        v[3] = 0.6406576735413505 * x[3]
        v[1] += v[3]
        v[3] = 0.5522910978804745 * x[4]
        v[1] += v[3]
        v[3] = 0.4761130154142021 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.793103448275862 * x[2]
        v[1] += x[1]
        v[2] = 2.4114149821640907 * x[3]
        v[1] += v[2]
        v[2] = 2.8826110131616716 * x[4]
        v[1] += v[2]
        v[2] = 3.230512342336356 * x[5]
        v[1] += v[2]
        v[2] = 0.896551724137931 * x[1]
        v[2] += x[0]
        v[3] = 0.8038049940546969 * x[2]
        v[2] += v[3]
        v[3] = 0.7206527532904179 * x[3]
        v[2] += v[3]
        v[3] = 0.6461024684672713 * x[4]
        v[2] += v[3]
        v[3] = 0.5792642820741053 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 1.8620689655172413 * x[2]
        v[2] += x[1]
        v[1] = 2.600475624256837 * x[3]
        v[2] += v[1]
        v[1] = 3.2281766370084872 * x[4]
        v[2] += v[1]
        v[1] = 3.7569297068633256 * x[5]
        v[2] += v[1]
        v[1] = 0.9310344827586207 * x[1]
        v[1] += x[0]
        v[3] = 0.8668252080856124 * x[2]
        v[1] += v[3]
        v[3] = 0.8070441592521218 * x[3]
        v[1] += v[3]
        v[3] = 0.7513859413726651 * x[4]
        v[1] += v[3]
        v[3] = 0.6995662212779985 * x[5]
        v[1] += v[3]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        v[1] = 1.9310344827586208 * x[2]
        v[1] += x[1]
        v[2] = 2.796670630202141 * x[3]
        v[1] += v[2]
        v[2] = 3.6003116158924113 * x[4]
        v[1] += v[2]
        v[2] = 4.34520367435291 * x[5]
        v[1] += v[2]
        v[2] = 0.9655172413793104 * x[1]
        v[2] += x[0]
        v[3] = 0.9322235434007136 * x[2]
        v[2] += v[3]
        v[3] = 0.9000779039731028 * x[3]
        v[2] += v[3]
        v[3] = 0.8690407348705821 * x[4]
        v[2] += v[3]
        v[3] = 0.839073812978493 * x[5]
        v[2] += v[3]
        v[3] = v[2] * v[2]
        v[2] = v[1] - v[3]
        v[1] = -1. + v[2]
        v[2] = v[1] * v[1]
        v[0] += v[2]
        v[2] = 2. * x[2]
        v[2] += x[1]
        v[1] = 3. * x[3]
        v[2] += v[1]
        v[1] = 4. * x[4]
        v[2] += v[1]
        v[1] = 5. * x[5]
        v[2] += v[1]
        v[1] = x[0] + x[1]
        v[1] += x[2]
        v[1] += x[3]
        v[1] += x[4]
        v[1] += x[5]
        v[3] = v[1] * v[1]
        v[1] = v[2] - v[3]
        v[2] = -1. + v[1]
        v[1] = v[2] * v[2]
        v[0] += v[1]
        return v[0]

class st_bsj3(BenchmarkFunction):
    @property
    def domain(self):
        return [[0, 99], [0, 99], [0, 99], [0, 99], [0, 99], [0, 99]]

    def _function(self,x):
        y = (10.5*x[0] - 1.5*x[0]**2 - x[1]**2 - 3.95*x[1] - x[2]**2 + 3*x[2]-
             2*x[3]**2 + 5*x[3] - x[4]**2 + 1.5*x[4] - 2.5*x[5]**2-1.5*x[5])
        return y
