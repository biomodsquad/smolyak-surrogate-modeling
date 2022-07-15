import numpy
from math
from BasisFunction import BasisFunction

class ChebyshevFirstKind(BasisFunction):
    """Basis Function of Chebyshev polynomials of the first kind
    Chebyshev polynomials are a set of polynomials described by the
    following recurrence relation:
    .. math::
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)

    The extrema of the polynomials are calculated via the following
    equation:
    .. math::
    extrema = -\cos\left((\frac{j-1}{n-1}\pi\right), j = 1,...,n

    """

    def update_extrema_index_per_level(self):
        """Compute extrema and extrema per level num"""
        next_exactness = len(self._extrema_per_level_num)

        for ex in range(next_exactness,self._max_exactness+1):
            m = 2**(ex)+1
            counter_index = 0
            for i in range(1,m+1):
                temp = round(-numpy.cos(numpy.pi*(i-1)/(m-1)),15) + 0
                if temp not in self._extrema_list:
                    self._extrema_list.append(temp)
                    counter_index += 1
            self._extrema_per_level_num.append(counter_index)

    def basis_fun(x,n):
        """Terms of basis function
        Returns the output of an nth degree Chebyshev polynomial of the
        first kind with input x

        Parameters
        ----------
        x : float
            input

        n : int
            degree of Chebyshev polynomial of the first kind
        
        Returns
        -------
        y : float
            output
        """
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            k_lim = math.floor(n/2)
            answer = 0
            for k in range(0,k_lim+1):
                answer += math.comb(n,2*k)*((x**2 - 1)**k)*(x**(n-2*k))
            return answer
