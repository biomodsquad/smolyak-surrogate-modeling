import numpy
from smolyak import IndexGrid
from basis import ChebyshevFirstKind


class SurrogateFunction:
    r"""Generate a surrogate function that approximates a complex function.

    Depending on dimensionality (number of independent variables),
    level of exactness (approximation's accuracy) and the basis function,
    a set of grid points by the Smolyak method are generated and a
    surrogate function is trained which acts as a black box of a
    complex function.

    :class:`IndexGrid` generates the required number of extremums
    of basis function and make the grid points based on the
    extremums' indexes (degrees).

    :class:`BasisFunction` generates the extremums depending on
    the type of basis function and level of exactness.

    This class uses the outputs of the mentioned classes to first
    generate the ``grid_points_basis_function`` which is grid points
    in the basis function domain.
    With ``domain_real_function`` (domain of the real function)
    being specified, the actual ``grid_points`` of the real function
    are generated. ``coeffcients_of_surrogate`` is obtained via solving
    linear equations (= number of grid points) using LU decomposition.

    For instance, consider function :math:F(y) with one independent variable.
    Basis function B(x, order) is chosen for the approximation and
    generates three grid points:
    (:math:x_0, :math:x_1, :math:x_2).
    The extremums' orders are: [:math:x_0: 0, :math:x_1: 1, :math:x_2: 2];
    then:
    ..math:
        Basis function matrix = \begin{bmatrix}
        B(x_0, 0) & B(x_0, 1) & B(x_0, 2) \\
        B(x_1, 0) & B(x_1, 1) & B(x_1, 2) \\
        B(x_2, 0) & B(x_2, 1) & B(x_2, 2)
        \end{bmatrix}

        Coefficients = \begin{bmatrix}
        C_0 \\
        C_1 \\
        C_2
        \end{bmatrix}

        Real function =\ begin{bmatrix}
        F(x_0) \\
        F(x_1) \\
        F(x_2)
        \end{bmatrix}

    ..math:
        (Basis function matrix) Coefficients = (Real function)

    Coefficients can be obtained by solving linear equations, since
    we have a full rank systems of matrixes.

    The generated surrogate is then:
    ..math:
        S(x) = C_0*B(x, 0) + C_1B(x, 1) + C_2B(x, 2)

    :meth:`~SurrogateFunction.output_value` evaluates
    the black box at a given input.

    Parameters
    ----------
    exactness: int
        Level of exactness of the approximation.
    domain_real_function : iterable
        Domain of the real function to be approximated.
    basis_function: :class:`BasisFunction`
        Basis function used for training.
    real_function: Callable
        Function that is being approximated.
    """

    def __init__(self, exactness, domain_real_function,
                 basis_function, real_function):
        self.exactness = exactness
        self.domain_real_function = domain_real_function
        self.basis_function = basis_function
        self.real_function = real_function

    @property
    def exactness(self):
        """int: Level of exactness."""
        return self._exactness

    @exactness.setter
    def exactness(self, value):
        self._exactness = value

    @property
    def basis_function(self):
        """:class:`BasisFunction`: Basis function chosen for the model."""
        return self._basis_function

    @basis_function.setter
    def basis_function(self, basis_function_instance):
        self._basis_function = basis_function_instance

    @property
    def domain_real_function(self):
        """numpy.ndarray: Domain of the real function."""
        return self._domain_real_function

    @domain_real_function.setter
    def domain_real_function(self, domain):
        self._domain_real_function = numpy.array(domain, dtype=float)

    @property
    def dimension(self):
        """int: Number of independent variables."""
        self._update()

        return self._dimension

    @property
    def grid_point_indexes(self):
        """numpy.ndarray: Indexes of grid points."""
        self._update()

        return self._grid_point_indexes

    @property
    def grid_points_basis_function(self):
        """numpy.ndarray: Grid points of the basis function."""
        self._update()

        return self._grid_points_basis_function

    @property
    def grid_points(self):
        """numpy.ndarray: Grid points of the real function."""
        self._update()

        return self._grid_points

    @property
    def coefficients(self):
        """numpy.ndarray: Coeffcients of the surrogate function."""
        self._update()

        return self._coefficients

    def output_value(self, *surrogate_inputs):
        """Generate the surrogate's output at a given input.

        With coefficients and grid points being computed,
        A surrogate function is constructed and can be
        evaluated at a given input.

        Parameters
        ----------
        surrogate_inputs: iterable
            Input of the surrogate function.

        Returns
        -------
        surrogate_output: float
            The output of the surrogate at a given input.

        Raises
        ------
        IndexError: Surrogate's input must be an array with a
        length of number of real function's independent variables.

        """
        self._update()

        if len(*surrogate_inputs) != self._dimension:
            raise IndexError("Inputs must be of length of {}".format(
                self._dimension))

        surrogate_output = 0
        for index_point in range(len(self._grid_points)):
            coefficient = self._coefficients[index_point]
            output_grid = 1
            for dimension_ in range(self._dimension):
                # compute basis function at input and indexes of grid points
                output_grid *= self._basis_function.basis(
                    surrogate_inputs[0][dimension_], self._grid_point_indexes[
                        index_point][dimension_])
            output_grid *= coefficient
            surrogate_output += output_grid

        return surrogate_output

    def _update(self):
        """Compute the following properties.

        Updates indexes of grid points (``grid_point_indexes``),
        grid points of the basis function (``grid_points_basis_function``),
        grid points of the real function (``grid_points``). Then, it
        computes the basis function matrix and real function
        at grid points, and then update the coefficients ``coefficients``.
        """
        self._dimension = len(self._domain_real_function)//2

        # get number of indexes per level to generate the grid points
        max_level = self._exactness + 1
        num_of_index_per_level = (
            self._basis_function(max_level)._num_extrema_per_level)

        # generate grid points indexes
        self._grid_point_indexes = (IndexGrid(self._dimension,
                                    self._exactness,
                                    num_of_index_per_level)
                                    .grid_point_indexes)

        # generate grid points of the basis function
        self._grid_points_basis_function = numpy.array(
            self._grid_point_indexes)
        extremums = numpy.array(
            self._basis_function(self._exactness)._extrema)
        self._grid_points_basis_function = extremums[self._grid_point_indexes]

        # transform linearly to real function domain
        self._grid_points = numpy.array(self._grid_points_basis_function)
        domain_real_function_ = numpy.array(
            self._domain_real_function).reshape((self._dimension, 2))
        for index in range(self._dimension):
            minimum = domain_real_function_[index][0]
            maximum = domain_real_function_[index][1]
            slope = (maximum-minimum) / (
                numpy.max(extremums)-numpy.min(extremums))
            intercept = minimum - slope*numpy.min(extremums)
            self._grid_points[:, index] = (
                slope*self._grid_points[:, index] + intercept)

        # compute basis matrix
        basis_matrix = numpy.ones((len(self._grid_points),
                                   len(self._grid_points)))
        for grid_num in range(len(self._grid_points)):
            for extrema_num in range(self._dimension):
                index_dimension = 0
                for index in numpy.transpose(
                        self._grid_point_indexes)[extrema_num]:
                    basis_matrix[grid_num][index_dimension] *= (
                        self._basis_function.basis(
                            self._grid_points[grid_num][extrema_num], index))
                    index_dimension += 1

        # evaluate the real function at grid points
        real_function_at_grids = numpy.zeros((len(self._grid_points)))
        for grid_num in range(len(self._grid_points)):
            real_function_at_grids[grid_num] = (
                self.real_function(*self._grid_points[grid_num]))

        # evaluate coefficients
        coefficients = numpy.linalg.solve(basis_matrix, real_function_at_grids)
        self._coefficients = coefficients


def branin(x1, x2):
    """Evaluate Branin function.

    Parameters
    ----------
    x1 : float
        Independent variable #1.
    x2 : float
        Independent variable #2.

    Returns
    -------
    branin : float
        Branin function's output.

    """
    branin = ((x2 - 5.1*x1**(2)/(4 * numpy.pi**2) + 5*x1 / (numpy.pi) - 6)**2
              + 10*(1 - 1/(8*numpy.pi))*numpy.cos(x1) + 10)

    return branin
