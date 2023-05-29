import abc
import math

import numpy
class test_fun(abc.ABC):
    """Abstract test function class

    Stores data about each test function including call, name, the number of
    dimensions, the domain, and how to normalize that domain to the interval
    [-1,1]

    Parameters
    ----------
    dim : int
        the number of variables used as input

    lower_bounds : list of floats of size 1xdim
        the lower bounds of the domain of each variable

    upper_bounds : list of floats of size 1xdim
        the upper bounds of the domain of each variable

    Raises
    ------
    ValueError
        The number of variables/dimensions should be greater than one

    IndexError
        All dimensions must have bounds
    """
    def __init__(self,dim,lower_bounds,upper_bounds):
        self._dim = dim
        if dim < 1:
            raise ValueError('The number of variables must be greater than 1')
        if not(hasattr(lower_bounds,'__len__') and hasattr(upper_bounds,'__len__')):
            if not dim == 1:
                raise IndexError('All dimensions must have bounds')
            # turn scalar inputs into 1x1 vectors
            if not hasattr(lower_bounds,'__len__'):
                self._lower_bounds = [lower_bounds]
            else:
                self._lower_bounds = [lower_bounds[0]]
            if not hasattr(lower_bounds,'__len__'):
                self._upper_bounds = [upper_bounds]
            else:
                self._upper_bounds = [upper_bounds[0]]
        else:
            if max(len(lower_bounds),len(upper_bounds)) < dim:
                raise IndexError('All dimensions must have bounds')
            self._upper_bounds = upper_bounds[0:dim]
            self._lower_bounds = lower_bounds[0:dim]
    
    @property
    def name(self):
        """Name of the function"""
        return self.__class__.__name__

    @property
    def dim(self):
        """Number of variables"""
        return self._dim

    @property
    def lower_bounds(self):
        """ the lower bounds of the domain of each variable"""
        return self._lower_bounds
    
    @property
    def upper_bounds(self):
        """the upper bounds of the domain of each variable"""
        return self._upper_bounds

    @property
    def bounds(self):
        """the lower and upper bounds of each variable"""
        return list(zip(self._lower_bounds,self._upper_bounds))

    def check_bounds(self,x):
        """Checks that the input of a call is within the bounds of the function

        Parameters
        ----------
        x : list of floats
            the input to the call method

        Raises
        ------
        ValueError
            if the input is outside is outside the domain
        """
        x = numpy.array(x,copy=False)
        for i in range(self.dim):
            if not (self.lower_bounds[i]-(1e-10) <= x[i] <= self.upper_bounds[i]+1e-10):
                print("input: " + str(x[i]))
                print("bounds: " + str(self.lower_bounds[i])
                      + " to " + str(self.upper_bounds[i]))
                raise ValueError("input out of bounds")
        
    @abc.abstractmethod
    def __call__(self,x):
        """Evaluate the test function

        Parameters
        ----------
        x : list of floats
            the input to the call method
        """
        pass

class beale(test_fun):
    def __init__(self):
        super().__init__(2,[-7.0000000008, -9.5000000002],
                         [11.69999999928, 9.44999999982])

    def __call__(self,x):
        self.check_bounds(x)
        y = (((-1.5) + x[0]*(1 - x[1]))**2 +
             ((-2.25) + (1 - (x[1])**2)*x[0])**2 +
             ((-2.625)+(1-pow(x[1],3))*x[0])**2)
        return y

class box2(test_fun):
    def __init__(self):
        super().__init__(2,[-10,0],[10,10])
    def __call__(self,x):
        self.check_bounds(x)
        term1 = [0.536957976864517,0.683395469841369,0.691031152313854,
                 0.652004407146905,0.599792712713548,0.546332883917360,
                 0.495673421825855,0.448993501489319,0.406446249936512,
                 0.36783404124168]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        y = 0
        for i in range(len(term1)):
            y += pow(math.exp(term2[i]*x[0])-math.exp(term2[i]*x[1])-term1[i],2)
        return y

class branin(test_fun):
    def __init__(self):
        super().__init__(2,[-5,0],[10,15])
    def __call__(self,x):
        self.check_bounds(x)
        y1 = pow(x[1] - 5.1*pow(x[0],2)/(4*pow(numpy.pi,2)) + 5*x[0]/numpy.pi - 6,2)
        y2 = 10*(1-1/(8*numpy.pi))*numpy.cos(x[0]) + 10
        return y1 + y2
    
class camel1(test_fun):
    def __init__(self):
        super().__init__(2,[-5,-5],[5,5])
    def __call__(self,x):
        self.check_bounds(x)
        y = (4*pow(x[0],2)-2.1*pow(x[0],4)+0.333333333333333*pow(x[0],6) +
                 x[0]*x[1]-4*pow(x[1],2)+4*pow(x[1],4))
        return y

class camel6(test_fun):
    def __init__(self):
        super().__init__(2,[-3,-1.5],[3,1.5])
    def __call__(self,x):
        self.check_bounds(x)
        y = (4*pow(x[0],2)-2.1*pow(x[0],4)+0.333333333333333*pow(x[0],6) +
             x[0]*x[1]-4*pow(x[1],2)+4*pow(x[1],4))
        return y

class chi(test_fun):
    def __init__(self):
        super().__init__(2,[-30,-30],[30,30])
    def __call__(self,x):
        self.check_bounds(x)
        y = (pow(x[0],2) - 12*x[0] + 10*math.cos(1.5707963267949*x[0]) +
             8*math.sin(15.707963267949*x[0]) -
             0.447213595499958*math.exp(-0.5*pow((-0.5)+x[1],2)) + 11)
        return y
    
class cliff(test_fun):
    def __init__(self):
        super().__init__(2,[-7, -6.8502133863], [11.7, 11.83480795233])
    def __call__(self,x):
        self.check_bounds(x)
        y = pow(-0.03+0.01*x[0],2)-x[0]+math.exp(20*x[0]-20*x[1])+x[1]
        return y
    
class cube(test_fun):
    def __init__(self):
        super().__init__(2,[-18, -18], [9.9, 9.9])
    def __call__(self,x):
        self.check_bounds(x)
        y = pow(-1+x[0],2)+100*pow(x[1]-pow(x[0],3),2)
        return y
    
class denschna(test_fun):
    def __init__(self):
        super().__init__(2,[-20, -20], [9, 9])
    def __call__(self,x):
        self.check_bounds(x)
        y = pow(x[0],4)+pow(x[0]+x[1],2)+pow(math.exp(x[1])-1,2)
        return y

class himmelp1(test_fun):
    def __init__(self):
        super().__init__(2,[0, 0], [95, 75])
    def __call__(self,x):
        self.check_bounds(x)
        y = (3.8112755343*x[0] -
             (0.1269366345*pow(x[0],2) - 0.0020567665*pow(x[0],3) +
              1.0345e-5*pow(x[0],4) +
              (0.0302344793*x[0] - 0.0012813448*pow(x[0],2) +
               3.52599e-5*pow(x[0],3) - 2.266e-7*pow(x[0],4))*x[1] +
              0.2564581253*pow(x[1],2) - 0.003460403*pow(x[1],3) +
              1.35139e-5*pow(x[1],4) - 28.1064434908/(1 +x[1]) +
              (3.405462e-4*x[0] - 5.2375e-6*pow(x[0],2)-
               6.3e-9*pow(x[0],3))*(x[1]**2) +
              (7e-10*pow(x[0],3) - 1.6638e-6*x[0])*pow(x[1],3) -
              2.8673112392*math.exp(5e-4*x[0]*x[1]))+ 6.8306567613*x[1] -
             75.1963666677)
        return y

class hs002(test_fun):
    def __init__(self):
        super().__init__(2,[-8.7756292513, 1.5], [11.2243707487, 11.5])

    def __call__(self,x):
        self.check_bounds(x)
        y = 100*pow(x[1]-pow(x[0],2),2)+pow(1-x[0],2)
        return y

class hs003(test_fun):
    def __init__(self):
        super().__init__(2,[-10, 0], [10, 10])

    def __call__(self,x):
        self.check_bounds(x)
        y = 1e-5*pow(x[1]-x[0],2)+x[1]
        return y

class hs004(test_fun):
    def __init__(self):
        super().__init__(2,[1, 0], [11, 10])

    def __call__(self,x):
        self.check_bounds(x)
        y = 0.333333333333333*pow(1+x[0],3)+x[1]
        return y

class hs3mod(test_fun):
    def __init__(self):
        super().__init__(2,[-10, 0], [10, 10])

    def __call__(self,x):
        self.check_bounds(x)
        y = pow(x[1]-x[0],2) + x[1]
        return y

class jensmp(test_fun):
    def __init__(self):
        super().__init__(2,[0.1, 0.1], [0.9, 0.9])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [x*2 for x in range(2,12)]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - math.exp((i+1)*x[0]) - math.exp((i+1)*x[1]),2)
        return y

class logros(test_fun):
    def __init__(self):
        super().__init__(2,[0, 0], [11, 11])

    def __call__(self,x):
        self.check_bounds(x)
        y = math.log(1+10000*pow(x[1]-x[0]**2,2)+pow(1-x[0],2))
        return y

class mdhole(test_fun):
    def __init__(self):
        super().__init__(2,[0, -10], [10, 10])
        
    def __call__(self,x):
        self.check_bounds(x)
        y = 100*pow(math.sin(x[0])-x[1],2)+x[0]
        return y

class median_vareps(test_fun):
    def __init__(self):
        super().__init__(2,[1e-08, -9.499789331], [10.00000001, 10.500210669])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [-0.171747132,-0.843266708,-0.550375356,-0.301137904,
                 -0.292212117,-0.224052867,-0.349830504,-0.856270347,
                 -0.067113723,-0.500210669,-0.998117627,-0.578733378,
                 -0.991133039,-0.762250467,-0.130692483,-0.639718759,
                 -0.159517864,-0.250080533,-0.668928609]
        y = x[0]
        for i in range(len(term1)):
            y += math.sqrt(x[0]**2 + pow(term1[i] + x[1],2))
        return y

class s328(test_fun):
    def __init__(self):
        super().__init__(2,[1, 1], [2.7, 2.7])

    def __call__(self,x):
        self.check_bounds(x)
        y = (0.1*(x[0]**2 + (1 + (x[1]**2))/(x[0]**2) +
                  (100 + (x[0]**2)*(x[1]**2))/(pow(x[0],4)*pow(x[1],4))) + 1.2)
        return y

class sim2bqp(test_fun):
    def __init__(self):
        super().__init__(2,[-10, 0], [9, 0.45])

    def __call__(self,x):
        self.check_bounds(x)
        y = pow(x[1]-x[0],2)+x[1]+pow(x[0]+x[1],2)
        return y

class simbqp(test_fun):
    def __init__(self):
        super().__init__(2,[-10, 0], [9, 0.45])

    def __call__(self,x):
        self.check_bounds(x)
        y = pow(x[1]-x[0],2)+x[1]+pow(2*x[0]+x[1],2)
        return y

class allinit(test_fun):
    def __init__(self):
        super().__init__(3,[-11.1426691153, 1, -1e10],
                         [8.8573308847, 11.2456257795, 1])
    def __call__(self,x):
        self.check_bounds(x)
        y = (x[0]**2 + x[1]**2 + pow(x[2] + 2,2) + x[2] + math.sin(x[2])**2 +
             pow(x[0],2)*pow(x[1],2) + math.sin(x[2])**2 + pow(x[1],4) +
             pow(-4 + math.sin(2)**2 + pow(x[1],2)*pow(x[2],2) + x[0],2) +
             pow(x[2]**2 + pow(x[0] + 2,2),2) + pow(math.sin(2),4) - 1)
        return y
    
class box3(test_fun):
    def __init__(self):
        super().__init__(3,[-9.0000004305, 3.23989999984065e-06, -8.9999997323],
                         [9.89999961255, 18.00000291591, 9.90000024093])
    def __call__(self,x):
        self.check_bounds(x)
        coeffs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        coeff2 = [0.536957976864517,0.683395469841369,0.691031152313854,
                  0.652004407146905,0.599792712713548,0.54633288391736,
                  0.495673421825855,0.448993501489319,0.406446249936512,
                  0.36783404124168]
        y = 0
        for i in range(10):
            y += pow(math.exp(-coeffs[i]*x[0])-math.exp(-coeffs[i]*x[1])
                     -coeff2[i]*x[2],2)
        return y

class eg1(test_fun):
    def __init__(self):
        super().__init__(3,[-10.2302657121, -1, 1], [9.7697342879, 1, 2])
    def __call__(self,x):
        self.check_bounds(x)
        y = x[0]**2 + pow(x[1]*x[2],4)+x[0]*x[2] + math.sin(x[0]+x[2])*x[1]+x[1]
        return y

class fermat_vareps(test_fun):
    def __init__(self):
        super().__init__(3,[-7.9999999999, -8.8452994616, 1e-08],
                         [12.0000000001, 11.1547005384, 10.00000001])

    def __call__(self,x):
        self.check_bounds(x)
        y = (math.sqrt(x[2]**2 + x[0]**2 + x[1]**2) +
             math.sqrt(x[2]**2 + pow(x[0] - 4,2) + x[1]**2)+
             math.sqrt(x[2]**2 + pow(x[0] - 2,2) + pow(x[1] - 4,2)) + x[2])
        return y

class fermat2_vareps(test_fun):
    def __init__(self):
        super().__init__(3,[-8, -9.00000002, 1e-08],
                         [12, 10.99999998, 10.00000001])

    def __call__(self,x):
        self.check_bounds(x)
        y = (math.sqrt(x[2]**2 + x[0]**2 + x[1]**2) +
             math.sqrt(x[2]**2 + pow(x[0] - 4,2) + x[1]**2)+
             math.sqrt(x[2]**2 + pow(x[0] - 2,2) + pow(x[1] - 1,2)) + x[2])
        return y

class least(test_fun):
    def __init__(self):
        super().__init__(3,[473.98605675534, -159.3518936954, -5],
                         [ 506.6511741726, -125.41670432586, 4.5])

    def __call__(self,x):
        self.check_bounds(x)
        term0 = [127,151,379,421,460,426]
        term1 = [-5,-3,-1,5,3,1]
        y = 0
        for i in range(len(term0)):
            y += pow(term0[i] + (-x[1]*math.exp(term1[i]*x[2])) - x[0],2)
        return y

class s242(test_fun):
    # is exactly the same as box3 with different bounds
    def __init__(self):
        super().__init__(3,[0, 0, 0], [10, 10, 10])

    def __call__(self,x):
        self.check_bounds(x)
        coeffs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        coeff2 = [0.536957976864517,0.683395469841369,0.691031152313854,
                  0.652004407146905,0.599792712713548,0.54633288391736,
                  0.495673421825855,0.448993501489319,0.406446249936512,
                  0.36783404124168]
        y = 0
        for i in range(10):
            y += pow(math.exp(-coeffs[i]*x[0])-math.exp(-coeffs[i]*x[1])
                     -coeff2[i]*x[2],2)
        return y

class s244(test_fun):
    def __init__(self):
        super().__init__(3,[0, 0, 0], [10, 10, 10])

    def __call__(self,x):
        self.check_bounds(x)
        y = (pow(0.934559787821252 + math.exp(-0.1*x[0]) -
                 math.exp(-0.1*x[1])*x[2],2) +
             pow(-0.142054336894918 + math.exp(-0.2*x[0]) -
                 math.exp(-0.2*x[1])*x[2],2) +
             pow(-0.491882878842398 + math.exp(-0.3*x[0]) -
                 math.exp(-0.3*x[1])*x[2],2))
        return y

class s333(test_fun):
    def __init__(self):
        super().__init__(3,[79.901992908, -1, -1], [89.9117936172, 0.9, 0.9])

    def __call__(self,x):
        self.check_bounds(x)
        coeff1 = [-4,-5.75,-7.5,-24,-32,-48,-72,-96]
        coeff2 = [0.013869625520111,0.0152439024390244,0.0178890876565295,
                  0.0584795321637427,0.102040816326531,0.222222222222222,
                  0.769230769230769,1.66666666666667]
        y = 0
        for i in range(8):
            y += pow(1-coeff2[i]*math.exp(coeff1[i]*x[1])*x[0]-coeff2[i]*x[2],2)
        return y

class st_cqpjk2(test_fun):
    def __init__(self):
        super().__init__(3,[0, 0, 0], [0.9, 0.9, 0.9])
        
    def __call__(self,x):
        self.check_bounds(x)
        y = 9*x[0]*x[0]-15*x[0]+9*x[1]*x[1]-12*x[1]+9*x[2]*x[2]-9*x[2]
        return y

class yfit(test_fun):
    def __init__(self):
        super().__init__(3,[-9.9978786299, -10.0035439984, 0],
                         [10.0021213701, 9.9964560016, 10010])

    def __call__(self,x):
        self.check_bounds(x)
        coeff1 = [x*0.0625 for x in range(17)]
        coeff0 = [x*0.0625 for x in reversed(range(17))]
        coeff_s = [-21.158931,-17.591719,-14.046854,-10.519732,-7.0058392,
                   -3.5007293,0,3.5007293,7.0058392,10.519732,14.046854,
                   17.591719,21.158931,24.753206,28.379405,32.042552,35.747869]
        y = 0
        for i in range(17):
            y += pow(coeff_s[i]+math.atan(coeff0[i]*x[0]+coeff1[i]*x[1])*x[2],2)
        return y

class brownden(test_fun):
    def __init__(self):
        super().__init__(4,[-21.5944399048, 3.2036300512,
                            -10.4034394882, -9.7632212255],
                         [-1.43499591432, 20.88326704608,
                          8.63690446062, 9.21310089705])
    def __call__(self,x):
        self.check_bounds(x)
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
        y = 0
        for i in range(len(coeff1)):
            y += pow(pow(coeff_s1[i] + x[0] + coeff1[i]*x[1],2) +
                     pow(coeff_s2[i] + x[2] + coeff3[i]*x[3],2),2)
        return y

class hatflda(test_fun):
    def __init__(self):
        super().__init__(4,[1e-07, 1e-07, 1e-07, 1e-07],
                         [10.999999997, 10.9999999714,
                          10.9999999281, 10.9999998559])
    def __call__(self,x):
        self.check_bounds(x)
        y = (pow(x[0] - 1, 2) + pow(x[0] - math.sqrt(x[1]), 2) +
             pow(x[1] - math.sqrt(x[2]), 2) + pow(x[2] - math.sqrt(x[3]), 2))
        return y

class hatfldb(test_fun):
    def __init__(self):
        super().__init__(4,[1e-07, 1e-07, 1e-07, 1e-07],
                         [10.9472135922, 0.8, 10.6400000036, 10.4096000079])

    def __call__(self,x):
        self.check_bounds(x)
        y = (pow(x[0] - 1, 2) + pow(x[0] - math.sqrt(x[1]), 2) +
             pow(x[1] - math.sqrt(x[2]), 2) + pow(x[2] - math.sqrt(x[3]),2))
        return y

class hatfldc(test_fun):
    def __init__(self):
        super().__init__(4,[0, 0, 0, -8.9999999978],
                         [10, 10, 10, 11.0000000022])

    def __call__(self,x):
        self.check_bounds(x)
        y = pow(x[0]-1,2)+pow(x[2]-x[1]**2,2)+pow(x[3]-x[2]**2,2)+pow(x[3]-1,2)
        return y

class himmelbf(test_fun):
    def __init__(self):
        super().__init__(4,[0, 0, 0, 0], [0.378, 0.378, 0.378, 0.378])

    def __call__(self,x):
        self.check_bounds(x)
        y = 1e4*(pow(0.135299688810716*(x[0]**2) - 1,2) +
                 pow((x[0]**2 + 4.28e-4*(x[1]**2) +
                      1.83184e-7*(x[2]**2))/(11.18 + 4.78504e-3*(x[3]**2))-1,2)+
                 pow((x[0]**2 +1e-3*(x[1]**2) +
                      1e-6*(x[2]**2))/(16.44 + 0.01644*(x[3]**2))-1,2) +
                 pow((x[0]**2 + 1.61e-3*(x[1]**2) +
                      2.5921e-6*(x[2]**2))/(16.2+ 0.026082*(x[3]**2))-1,2) +
                 pow((x[0]**2 + 2.09e-3*(x[1]**2) +
                      4.3681e-6*(x[2]**2))/(22.2 + 0.046398*(x[3]**2))-1,2) +
                 pow((x[0]**2 + 3.48e-3*(x[1]**2) +
                      1.21104e-5*(x[2]**2))/(24.02+0.0835896*(x[3]**2))-1,2) +
                 pow((x[0]**2+0.00525*(x[1]**2)+
                      2.75625e-5*(x[2]**2))/(31.32+0.16443*(x[3]**2))-1,2))
        return y

class kowalik(test_fun):
    def __init__(self):
        super().__init__(4,[0, 0, 0, 0], [0.378, 0.378, 0.378, 0.378])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,
                 0.0235,0.0246]
        term2 = [16,4,1,0.25,0.0625,0.0277777777777778,0.015625,0.01,
                 0.00694444444444444,0.00510204081632653,0.00390625]
        coeff = [4,2,1,0.5,0.25,0.166666666666667,0.125,0.1,0.0833333333333333,
                 0.0714285714285714,0.0625]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - x[0]*(term2[i] + coeff[i]*x[1])/
                     (term2[i] + coeff[i]*x[2] + x[3]),2)
        return y

class palmer1(test_fun):
    def __init__(self):
        super().__init__(4,[1.3636340716, 1e-05, 1e-05, 1e-05],
                         [21.3636340716, 160.4544000091, 11.5013647921,
                          10.0931561774])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [78.596218,65.77963,43.96947,27.038816,14.6126,6.2614,1.53833,
                 0,1.188045,4.6841,16.9321,33.6988,52.3664,70.163,83.4221,
                 88.3995,78.596218,65.77963,43.96947,27.038816,14.6126,6.2614,
                 1.53833,0,1.188045,4.6841,16.9321,33.6988,52.3664,70.163,
                 83.4221]
        term2 = [3.200388615369,3.046173318241,2.749172911969,2.467400073616,
                 2.2008612609,1.949550365169,1.713473146009,1.485015206544,
                 1.287008567296, 1.096623651204,0.761544202225,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089,0,3.200388615369,
                 3.046173318241,2.749172911969,2.467400073616,2.2008612609,
                 1.949550365169,1.713473146009,1.485015206544,1.287008567296,
                 1.096623651204,0.761544202225,0.487388289424,0.274155912801,
                 0.121847072356,0.030461768089]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - x[1]/(term2[i]/x[3] + x[2]) -term2[i]*x[0],2)
        return y

# caused RuntimeWarning: divide by zero encountered in double_scalars
# fixed when LU for x's changed from 0 to 0.000001
class palmer3(test_fun):
    def __init__(self):
        super().__init__(4,[0.000001, 0.000001, 0.000001, 7.3225711014],
                         [10.0375049888, 10.0034428969,
                          14.6439962785, 27.3225711014])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [64.87939,50.46046,28.2034,13.4575,4.6547,0.59447,0,0.2177,
                 2.3029,5.5191,8.5519,9.8919,8.5519,5.5191,2.3029,0.2177,
                 0,0.59447,4.6547,13.4575,28.2034,50.46046,64.87939]
        term2 = [2.749172911969,2.467400073616,1.949550365169,1.4926241929,
                 1.096623651204,0.761544202225,0.587569773961,0.487388289424,
                 0.274155912801,0.121847072356,0.030461768089,0,0.030461768089,
                 0.121847072356,0.274155912801,0.487388289424,0.587569773961,
                 0.761544202225,1.096623651204,1.4926241929,1.949550365169,
                 2.467400073616,2.749172911969]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - x[0]/(term2[i]/x[2] + x[1]) -term2[i]*x[3],2)
        return y


# caused RuntimeWarning: divide by zero encountered in double_scalars
# fixed when LU for x's changed from 0 to 0.000001
class palmer4(test_fun):
    def __init__(self):
        super().__init__(4,[0.00001, 0.00001, 0.00001, 8.2655580306],
                         [19.3292787916, 10.8767116668,
                          10.0158603779, 28.2655580306])

    def __call__(self,x):
        self.check_bounds(x)
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
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - x[0]/(term2[i]/x[2] + x[1]) - term2[i]*x[3],2)
        return y

class palmer5d(test_fun):
    def __init__(self):
        super().__init__(4,[70.2513178169, -142.1059487487, 41.6401308813,
                            -9.304685674],
                         [81.22618603521, -109.89535387383, 55.47611779317,
                          9.6257828934])

    def __call__(self,x):
        self.check_bounds(x)
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
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - x[0] - term2[i]*x[1] - term3[i]*x[2] -
                     term4[i]*x[3],2)
        return y

class s257(test_fun):
    def __init__(self):
        super().__init__(4,[0, -9, 0, -9], [11, 11, 11, 11])

    def __call__(self,x):
        self.check_bounds(x)
        y = (100*pow(x[0]**2 - x[1],2) + pow(x[0] - 1,2) +
             90*pow(x[2]**2 - x[3],2) + pow(x[2] - 1,2)+
             10.1*(pow(x[1] - 1,2)+pow(x[3] - 1,2))+ (19.8*x[0]-19.8)*(x[3]-1))
        return y

class s351(test_fun):
    def __init__(self):
        super().__init__(4,[-7.3, 80, 1359, 0], [11.43, 90, 1490, 18])

    def __call__(self,x):
        self.check_bounds(x)
        term0 = [0.135299688810716,0.0894454382826476,0.0608272506082725,
                 0.0617283950617284,0.045045045045045,0.0416319733555371,
                 0.0416319733555371]
        term1 = [0,4.28e-4,0.001,0.00161,0.00209,0.00348,0.00525]
        term2 = [0,1.83184e-7,1e-6,2.5921e-6,4.3681e-6,1.21104e-5,2.75625e-5]
        y = 0
        for i in range(len(term0)):
            y += pow(-1 +
                     term0[i]*(x[0]**2 +term1[i]*(x[1]**2) +
                               term2[i]*(x[2]**2))/(1+term1[i]*(x[3]**2)),2)
        y = y * 10000
        return y

class s352(test_fun):
    def __init__(self):
        super().__init__(4,[-20.2235736001, 1.9084286837,
                            -10.4580411955, -9.4196803043],
                         [-0.20121624009, 19.71758581533,
                          8.58776292405, 9.52228772613])

    def __call__(self,x):
        self.check_bounds(x)
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
        y = 0
        for i in range(20):
            y += (pow(term1[i] + x[0] + term2[i]*x[1],2) +
                  pow(term3[i] + x[2] + term4[i]*x[3],2))
        return y

class shekel(test_fun):
    def __init__(self):
        super().__init__(4,[0, 0, 0, 0], [10, 10, 10, 10])
    def __call__(self,x):
        self.check_bounds(x)
        y = -(1/(0.1 + pow(x[0] - 4,2) + pow(x[1] - 4,2) +
                 pow(x[2] - 4,2) + pow(x[3] - 4,2)) +
              1/(0.2 + pow(x[0] - 1,2) + pow(x[1] - 1,2) +
                 pow(x[2] - 1,2) + pow(x[3] - 1,2)) +
              1/(0.2 + pow(x[0] - 8,2) + pow(x[1] - 8,2) +
                 pow(x[2] - 8,2) + pow(x[3] - 8,2)) +
              1/(0.4 + pow(x[0] - 6,2) + pow(x[1] - 6,2) +
                 pow(x[2] - 6,2) + pow(x[3] - 6,2)) +
              1/(0.4 + pow(x[0] - 3,2) + pow(x[1] - 7,2) +
                 pow(x[2] - 3,2) + pow(x[3] - 7,2)))
        return y

class hs045(test_fun):
    def __init__(self):
        super().__init__(5,[0, 0, 0, 0, 0], [1, 2, 3, 4, 5])

    def __call__(self,x):
        self.check_bounds(x)
        y = -(0.00833333333333333*x[0]*x[1]*x[2]*x[3]*x[4] - ( 2 ))
        return y

class s267(test_fun):
    def __init__(self):
        super().__init__(5,[-8.2232795288, 6.1236156871,
                            -10.5942083977, -5.2928287845, -8.2232796262],
                         [11.7767204712, 26.1236156871, 9.4057916023,
                          14.7071712155, 0])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] + math.exp(term2[i]*x[0])*x[2] -
                     math.exp(term2[i]*x[1])*x[3] + 3*math.exp(term2[i]*x[4]),2)
        return y

class s358(test_fun):
    def __init__(self):
        super().__init__(5,[-0.5, 1.5, -2, 0.001, 0.001],
                         [0.45, 2.25, -0.9, 0.09, 0.09])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [0.844,0.908,0.932,0.936,0.925,0.908,0.881,0.85,0.818,0.784,
                 0.751,0.718,0.685,0.658,0.628,0.603,0.58,0.558,0.538,
                 0.522,0.506,0.49,0.478,0.467,0.457,0.448,0.438,0.431,0.424,
                 0.420,0.414,0.411,0.406]
        term2 = [round(x*-10,ndigits=1) for x in range(len(term1))]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] - math.exp(term2[i]*x[3])*x[1] -
                     math.exp(term2[i]*x[4])*x[2] - x[0],2)
        return y

class biggs6(test_fun):
    def __init__(self):
        super().__init__(6,[-8.2885839998, 7.6831983277, -3.05828543,
                            -4.8134383873, -8.2885839966, -14.6154272503],
                         [10.54027440018, 24.91487849493, 15.247543113,
                          13.66790545143, 10.54027440306, 4.84611547473])

    def __call__(self,x):
        self.check_bounds(x)
        term1 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334,-0.325852731997495,
                 -0.28907018464926]
        term2 = [round(x*-0.1,ndigits=1) for x in range(1,len(term1)+1)]
        y = 0
        for i in range(len(term1)):
            y += pow(term1[i] + math.exp(term2[i]*x[0])*x[2] -
                     math.exp(term2[i]*x[1])*x[3] +
                     math.exp(term2[i]*x[4])*x[5],2)
        return y

class hart6(test_fun):
    def __init__(self):
        super().__init__(6,[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
    def __call__(self,x):
        self.check_bounds(x)
        y = -(math.exp(-(10*pow(x[0]-0.1312,2) + 0.05*pow(x[1]-0.1696,2) +
                         17*pow(x[2]-0.5569,2) + 3.5*pow(x[3]-0.0124,2) +
                         1.7*pow(x[4]-0.8283,2) + 8*pow(x[5]-0.5886,2))) +
              1.2*math.exp(-(0.05*pow(x[0]-0.2329,2)+ 10*pow(x[1]-0.4135,2) +
                             17*pow(x[2]-0.8307,2) + 0.1*pow(x[3]-0.3736,2) +
                             8*pow(x[4]-0.1004,2) + 14*pow(x[5]-0.9991,2))) +
              3*math.exp(-(3*pow(x[0]-0.2348,2) + 3.5*pow(x[1]-0.1451,2) +
                             1.7*pow(x[2]-0.3522,2) + 10*pow(x[3]-0.2883,2) +
                             17*pow(x[4]-0.3047,2) + 8*pow(x[5]-0.665,2))) +
              3.2*math.exp(-(17*pow(x[0]-0.4047,2) + 8*pow(x[1]-0.8828,2) +
                             0.05*pow(x[2]-0.8732,2) + 10*pow(x[3]-0.5743,2) +
                             0.1*pow(x[4]-0.1091,2) + 14*pow(x[5]-0.0381,2))))
        return y
    
# caused RuntimeWarning: invalid value encountered in double_scalars
# trying to not do all terms at once causes deviation for unknown reasons
##class palmer2a(test_fun):
##    def __init__(self):
##        super().__init__(6,[0, 0, -20.7797273226, -25.3729423234,
##                            3.6520539577,-10.081937131],
##                         [32.4286981517, 10.7435278989, -0.779727322599999,
##                          -5.3729423234, 23.6520539577, 9.918062869])
##
##    def __call__(self,x):
##        self.check_bounds(x)
##        y = (pow(72.676767 - x[0]/(3.046173318241 + x[1]) - x[2] -
##                 3.046173318241*x[3] -9.27917188476338*x[4] -
##                 28.2659658107383*x[5],2) +
##             pow(40.149455 - x[0]/(2.467400073616 + x[1]) - x[2] -
##                 2.467400073616*x[3] - 6.08806312328024*x[4] -
##                 5.0216873985605*x[5],2) +
##             pow(18.8548 - x[0]/(1.949550365169 + x[1]) - x[2] -
##                 1.949550365169*x[3] - 3.80074662633058*x[4] -
##                 7.40974697327763*x[5],2) +
##             pow(6.4762 - x[0]/(1.4926241929 + x[1]) - x[2] -
##                 1.4926241929*x[3] - 2.22792698123038*x[4] -
##                 3.32545771219912*x[5],2) +
##             pow(0.8596 - x[0]/(1.096623651204 + x[1]) - x[2] -
##                 1.096623651204*x[3] - 1.20258343237999*x[4] -
##                 1.31878143449399*x[5],2) +
##             pow((-x[0]/(0.878319472969 + x[1])) - x[2] -
##                 0.878319472969*x[3] - 0.771445096596542*x[4] -
##                 0.677575250667194*x[5],2) +
##             pow(0.273 - x[0]/(0.761544202225 + x[1]) - x[2] -
##                 0.761544202225*x[3] - 0.579949571942512*x[4] -
##                 0.44165723409569*x[5],2) +
##             pow(3.2043 - x[0]/(0.487388289424 + x[1]) - x[2] -
##                 0.487388289424*x[3] -0.237547344667653*x[4] -
##                 0.115777793974781*x[5],2) +
##             pow(8.108 - x[0]/(0.274155912801 + x[1]) - x[2] -
##                 0.274155912801*x[3] - 0.0751614645237495*x[4] -
##                 0.0206059599139685*x[5],2) +
##             pow(13.4291 - x[0]/(0.121847072356 + x[1]) - x[2] -
##                 0.121847072356*x[3] - 0.0148467090417283*x[4] -
##                 0.00180902803085595*x[5],2) +
##             pow(17.714 - x[0]/(0.030461768089 + x[1]) - x[2] -
##                 0.030461768089*x[3] -0.000927919315108019*x[4] -
##                 2.82660629821242e-5*x[5],2) +
##             pow(19.4529 - x[0]/x[1] -x[2],2) +
##             pow(17.7149 - x[0]/(0.030461768089 + x[1]) - x[2] -
##                 0.030461768089*x[3] - 0.000927919315108019*x[4] -
##                 2.82660629821242e-5*x[5],2) +
##             pow(13.4291 - x[0]/(0.121847072356 + x[1]) - x[2] -
##                 0.121847072356*x[3] - 0.0148467090417283*x[4] -
##                 0.00180902803085595*x[5],2) +
##             pow(8.108 - x[0]/(0.274155912801 + x[1]) - x[2] -
##                 0.274155912801*x[3] - 0.0751614645237495*x[4] -
##                 0.0206059599139685*x[5],2) +
##             pow(3.2053 - x[0]/(0.487388289424 + x[1]) - x[2] -
##                 0.487388289424*x[3] -0.237547344667653*x[4] -
##                 0.115777793974781*x[5],2) +
##             pow(0.273 - x[0]/(0.761544202225 + x[1]) - x[2] -
##                 0.761544202225*x[3] - 0.579949571942512*x[4] -
##                 0.44165723409569*x[5],2) +
##             pow((-x[0]/(0.878319472969 + x[1])) - x[2] -
##                 0.878319472969*x[3] - 0.771445096596542*x[4] -
##                 0.677575250667194*x[5],2) +
##             pow(0.8596 - x[0]/(1.096623651204 + x[1]) - x[2] -
##                 1.096623651204*x[3] -1.20258343237999*x[4] -
##                 1.31878143449399*x[5],2) +
##             pow(6.4762 - x[0]/(1.4926241929+ x[1]) - x[2] -
##                 1.4926241929*x[3] - 2.22792698123038*x[4] -
##                 3.32545771219912*x[5],2)+
##             pow(18.8548 - x[0]/(1.949550365169 + x[1]) - x[2] -
##                 1.949550365169*x[3] -3.80074662633058*x[4] -
##                 7.40974697327763*x[5],2) +
##             pow(40.149455 - x[0]/(2.467400073616 + x[1]) - x[2] -
##                 2.467400073616*x[3] - 6.08806312328024*x[4] -
##                 15.0216873985605*x[5],2) +
##             pow(72.676767 - x[0]/(3.046173318241 + x[1]) - x[2] -
##                 3.046173318241*x[3]-9.27917188476338*x[4]-
##                 28.2659658107383*x[5],2))
##        return y

class palmer5c(test_fun):
    def __init__(self):
        super().__init__(6,[27.5370157298, -11.7302338172, 30.7938174564,
                            -9.1697871977, -6.2910484684, -10.1772297675],
                         [42.78331415682, 7.44278956452, 45.71443571076,
                          9.74719152207, 12.33805637844, 8.84049320925])

    def __call__(self,x):
        self.check_bounds(x)
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
        y = 0
        for i in range(len(term1)):
            y += pow(term0[i] - x[0] + term1[i]*x[1] + term2[i]*x[2] +
                     term3[i]*x[3] + term4[i]*x[4] + term5[i]*x[5],2)
        return y

class palmer6a(test_fun):
    def __init__(self):
        super().__init__(6,[-44.1581372624, 0.120997701, -1.1808208888,
                            -8.633061426, 1e-05, 1e-05],
                         [-24.1581372624, 20.120997701, 18.8191791112,
                          11.366938574, 43.2710391882, 10.7437425261])

    def __call__(self,x):
        self.check_bounds(x)
        term0 = [10.678659,75.414511,41.513459,20.104735,7.432436,1.298082,
                 0.1713,0,0.068203,0.774499,2.070002,5.574556,9.026378]
        term1 = [0,2.467400073616,1.949550365169,1.4926241929,1.096623651204,
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
        term5 = [0,2.467400073616,1.949550365169,1.4926241929,1.096623651204,
                 0.761544202225,0.616850018404,0.536979718521,0.487388289424,
                 0.373156048225,0.274155912801,0.121847072356,0.030461768089]
        y = 0
        for i in range(len(term1)):
            y += pow(term0[i] - x[4]/(term5[i] + x[5])- x[0] -term1[i]*x[1] -
                     term2[i]*x[2] - term3[i]*x[3],2)
        return y

class palmer8a(test_fun):
    def __init__(self):
        super().__init__(6,[1e-05, 1e-05, -17.7129671187, -5.0299734848,
                            2.8287670723, -9.0495003432],
                         [12.4961104793, 10.2011908033, 2.2870328813,
                          14.9700265152, 22.8287670723, 10.9504996568])

    def __call__(self,x):
        self.check_bounds(x)
        term0 = [4.757534,3.121416,1.207606,0.131916,0,0.258514,
                 3.380161,10.762813,23.745996,44.471864,76.541947,97.874528]
        term1 = [0,0.030461768089,0.098695877281,0.190385614224,0.264714366016,
                 0.373156048225,0.616850018404,0.921467524761,1.287008567296,
                 1.713473146009,2.2008612609,2.467400073616]
        term3 = [0,0.030461768089,0.098695877281,0.190385614224,0.264714366016,
                 0.373156048225,0.616850018404,0.921467524761,1.287008567296,
                 1.713473146009,2.2008612609,2.467400073616]
        term4 = [0,9.27919315108019e-4,0.00974087619226621,0.0362466821034498,
                 0.0700736955752528,0.139245436326899,0.380503945205015,
                 0.849102399189164,1.6563910522933,2.93599022209398,
                 4.84379028973034,6.08806312328024]
        term5 = [0,2.82660629821242e-5,9.61384321281321e-4,0.00690084683584735,
                 0.0185495138986012,0.0519602767531113,0.234713865602508,
                 0.782420286049465,2.13178947509392,5.03074040250303,
                 10.6605104045911,15.0216873985605]
        y = 0
        for i in range(len(term1)):
            y += pow(term0[i] - x[0]/(term1[i] + x[1]) - x[2] - term3[i]*x[3] -
                     term4[i]*x[4] - term5[i]*x[5],2)
        return y

class s272(test_fun):
    def __init__(self):
        super().__init__(6,[0, 0, 0, 0, 0, 0],
                         [10.999999907, 20.0000005284, 13.9999995189,
                          10.9999998493, 14.9999995532, 12.99999966])

    def __call__(self,x):
        self.check_bounds(x)
        term0 = [-1.07640035028567,-1.49004122924658,-1.395465514579,
                 -1.18443140557593,-0.978846774427044,-0.808571735078932,
                 -0.674456081839291,-0.569938262912808,-0.487923778062043,
                 -0.422599358188832,-0.369619594903334,-0.325852731997496,
                 -0.28907018464926]
        term1 = [round(x*-0.1,ndigits=1) for x in range(1,len(term0)+1)]
        y = 0
        for i in range(len(term1)):
            y += pow(term0[i] + math.exp(term1[i]*x[0])*x[3] -
                     math.exp(term1[i]*x[1])*x[4] +
                     math.exp(term1[i]*x[2])*x[5],2)
        return y

class st_bsj3(test_fun):
    def __init__(self):
        super().__init__(6,[0, 0, 0, 0, 0, 0], [99, 99, 99, 99, 99, 99])

    def __call__(self,x):
        self.check_bounds(x)
        y = (10.5*x[0] - 1.5*(x[0]**2) - x[1]**2 - 3.95*x[1] - x[2]**2 + 3*x[2]-
             2*(x[3]**2) + 5*x[3] - x[4]**2 + 1.5*x[4] - 2.5*(x[5]**2)-1.5*x[5])
        return y
