import numpy

from smolyay.basis import (ChebyshevFirstKind, BasisFunctionSet,
                           NestedBasisFunctionSet)
from smolyay.grid import (IndexGridGenerator, SmolyakGridGenerator,
                          TensorGridGenerator, generate_compositions)


def make_slow_nested_set(exactness):
    """Create a slow growth nested set of Chebyshev polynomials.

    A nested set is created up to a given level of ``exactness``,
    which corresponds to a highest-order Chebyshev polynomial of
    degree ``n = 2**exactness``.
    
    Each nesting level corresponds to the increasing powers of 2 going up to
    ``2**exactness``, with the first level being a special case. The 
    generating Chebyshev polynomials are hence of degree (0, 2, 4, ...).
    Each new point added in a level is paired with a basis function of 
    increasing order. However, if the sum of the current points exceed a
    precision rule, then no points would be added.
    
    The precision rule is defined as 2 * ``exactness`` + 1. For a given
    ``exactness``, points are added at levels where the precision rule
    is greater than the sum of all points added at previous levels.
    
    For example, for an ``exactness`` of 3, the generating polynomials are
    of degree 0, 2, 4, and 8, at each of 4 levels. There are 1, 2, 2, and 4
    new points added at each level. The polynomial at level 0 is of degree 
    0, the polynomials at level 1 are of degrees 1 and 2, those at level 2 
    are of degree 3 and 4, and those at level 3 are of degrees 5, 6, 7, and
    8.
    
    For an ``exactness`` of 4, there would be 1, 2, 2, 4, and 8 new points
    added at each level. However, the precision rule states that for level
    4 the precision requirement is 9. The precision before points are
    added is 1 + 2 + 2 + 4 = 9, meaning that instead of adding 8 points to
    the new level, 0 are added.
    
    Parameters
    ----------
    exactness : int
        Level of exactness.
        
    Returns
    -------
    NestedBasisFunctionSet
        Nested Chebyshev polynomials of the first kind.
    """
    basis_functions = []
    levels = []
    points = []
    rule_add = -1 # tracks quadrature rules added
    precision_has = 0 # tracks precision variable
    for i in range(0, exactness+1):
        if 2 * i + 1 > precision_has:
            rule_add = rule_add + 1
            if rule_add > 1:
                start_level = 2**(rule_add-1)+1
                end_level = 2**rule_add
            elif rule_add == 1:
                start_level = 1
                end_level = 2
            else:
                start_level = 0
                end_level = 0
            level_range = range(start_level, end_level+1)
            precision_has = precision_has + len(level_range)

            basis_functions.extend(ChebyshevFirstKind(n) for n in level_range)
            levels.append(list(level_range))
            for p in basis_functions[end_level].points:
                if not numpy.isclose(points, p).any():
                    points.append(p)
        else:
            levels.append([])
    return NestedBasisFunctionSet(points,basis_functions,levels)
