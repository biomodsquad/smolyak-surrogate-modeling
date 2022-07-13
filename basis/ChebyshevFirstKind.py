import numpy
from BasisFunction import BasisFunction

class ChebyshevFirstKind(BasisFunction):
    """Basis Function of Chebyshev polynomials of the first kind"""

    def update_extrema_index_per_level(self, next_exactness = 1):
        """Compute extrema and extrema per level num

        Parameters
        ----------
        next_exactness : int
            the next level of exactness to compute
        """

        for ex in range(next_exactness,self._max_exactness+1):
            m = 2**(ex)+1
            counter_index = 0
            for i in range(1,m+1):
                temp = round(-np.cos(np.pi*(i-1)/(m-1)),15) + 0
                if temp not in self._extrema_list:
                    self._extrema_list.append(temp)
                    counter_index += 1
            self._index_per_level.append(counter_index)

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
            k_lim = floor(n/2)
            answer = 0
            for k in rnage(0,k_lim+1):
                answer += comb(n,2*k)*((x**2 - 1)**k)*(x**(n-2*k))
            return answer
