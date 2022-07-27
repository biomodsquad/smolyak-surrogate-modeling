import numpy


class SurrogateFunction:
    r"""Generate a surrogate function that approximates a complex function.

    Depending on dimensionality (number of independent variables),
    level of exactness (approximation's accuracy) and the basis function,
    a set of grid points can be generated.
    For instance, :class:`SmolyakGrid` generates grid points using
    Smolyak's method depending on the mentioned parameters.
    A surrogate function is then trained which acts as a black box of a
    complex function (``real_function``).

    The grid generator (``basis_grid``) calls the required number of points
    of a basis function based on the type of the basis function
    and exactness, and makes grid points based on the points' indexes
    (order of the points) via :meth:`~GridGenerator.generate_grid_index`.
    Then, :meth:`~GridGenerator.generate_grid_basis` generates the grid points
    of the basis function (bound between (-1, 1)).
    In case of using :class:`SmolyakGrid` the methods would be
    :meth:`~SmolyakGrid.generate_grid_index` and
    :meth:`~Smolyak.generate_grid_basis`

    With ``domain_real_function`` (domain of the ``real_function``
    to be approximated) being specified, the actual grid points
    of the real function are generated (property ``grids``).
    Coefficent of the surrogate(property `coeffcients_of_surrogate``)
    is obtained via solving linear equations (= number of grid points).

    For instance, consider a function :math:F(y) with one independent variable.
    Basis function B(x, order) is chosen for the approximation and
    we have three grid points:
    (:math:x_0, :math:x_1, :math:x_2).
    The point' orders are: [:math:x_0: 0, :math:x_1: 1, :math:x_2: 2];
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

    Coefficients (``coefficients_of_surrogate``) can be obtained
    by solving linear equations, since we have a full rank systems of matrixes.

    The generated surrogate is then:
    ..math:
        S(x) = C_0*B(x, 0) + C_1B(x, 1) + C_2B(x, 2)

    :meth:`~SurrogateFunction.output_value` evaluates
    the black box at a given input.

    Parameters
    ----------
    real_function: Callable
        Function that is being approximated.
    domain_real_function : iterable
        Domain of the real function to be approximated.
    basis_grids: :class:`PointsGrid(basis: :class:`BasisFunction`, exactness)`
        Grid generator.
        basis function and level of exactness.
    solve_method: Callabe
        Method that is being used for solving linear equations.
    """

    def __init__(self, real_function, domain_real_function,
                 basis_grids, solve_method):

        self.real_function = real_function
        self.domain_real_function = domain_real_function
        self.basis_grids = basis_grids
        self.solve_method = solve_method
        self._grids = []
        self._dimension = len(domain_real_function)

    @property
    def real_function(self):
        """Callable: Any callable function."""
        return self._real_function

    @real_function.setter
    def real_function(self, callable_function):
        self._real_function = callable_function

    @property
    def domain_real_function(self):
        """list: domain of the called function."""
        return list(self._domain_real_function)

    @domain_real_function.setter
    def domain_real_function(self, domain):
        self._domain_real_function = domain

    @property
    def grids(self):
        """list: Grid points of real function."""
        self._update()
        return self._grids.tolist()

    @property
    def coeffcients_of_surrogate(self):
        """list: Coefficient of the surrogate function."""
        self._update()
        return self._coefficients.tolist()

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
        basis = self.basis_grids.basis
        exactness = self.basis_grids.exactness

        if len(*surrogate_inputs) != self._dimension:
            raise IndexError("Inputs must be of length of {}".format(
                self._dimension))

        surrogate_output = 0
        for index_point in range(len(self._grids)):
            coefficient = self._coefficients[index_point]
            output_grid = 1
            for dimension_ in range(self._dimension):
                # compute basis function at input and indexes of grid points
                output_grid *= basis(exactness).basis(
                    surrogate_inputs[0][dimension_],
                    self.basis_grids.generate_grid_index(
                            self._dimension)[
                        index_point][dimension_])
            output_grid *= coefficient
            surrogate_output += output_grid

        return surrogate_output

    def _update(self):
        """Update properties of the surrogate.

        Update grid points of the real function ``grids``
        based on ``real_function`` and its domain ``domain_real_function``.
        Then, it updates the ``basis_matrix``, real function value at
        grid points (``real_function_at_grids``), and generate the
        coefficinets accordingly (``coefficients``).
        """
        domain_basis = (-1, 1)
        basis = self.basis_grids.basis
        exactness = self.basis_grids.exactness

        # create grid points of the real function
        self._grids = numpy.array(
            self.basis_grids.generate_grid_basis(self._dimension))
        for dimension_ in range(self._dimension):
            self._grids[:, dimension_] = (
                numpy.polynomial.polyutils.mapdomain(
                    self._grids[:, dimension_],
                    domain_basis, self.domain_real_function[dimension_]))

        # create basis matrix
        basis_matrix = numpy.ones((len(self._grids),
                                   len(self._grids)))
        for grid_num in range(len(self._grids)):
            for point_num in range(self._dimension):
                dimension_ = 0
                for index in numpy.transpose(
                        self.basis_grids.generate_grid_index(
                                self._dimension))[point_num]:
                    basis_matrix[grid_num][dimension_] *= (
                        basis(
                            exactness).basis(
                            self._grids[grid_num][point_num], index))
                    dimension_ += 1

        # evaluate the real function at grid points
        real_function_at_grids = numpy.zeros((len(self._grids)))
        for grid_num in range(len(self._grids)):
            real_function_at_grids[grid_num] = (
                self.real_function(*self._grids[grid_num]))

        # evaluate coefficients
        self._coefficients = self.solve_method(
                basis_matrix, real_function_at_grids)


# real function
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


# numerical methods for solving linear equations
def lower_upper_decomposition(a, b):
    """Solve systems of linear equations (Ax=B) via LU decompostion method.

    Parameters
    ----------
    a: numpy.ndarray
        A matrix (square matrix).
    b: numpy.ndarray
        B matrix.

    Returns
    -------
    x: numpy.ndarray
        Unknown variables.
    """
    x = numpy.linalg.solve(a, b)

    return x


def inverse_matrix(a, b):
    """Solve( Ax=B) by inversing the A matrix (x = :math:A^(-1) B).

    Parameters
    ----------
    a: numpy.ndarray
        A matrix(square matrix).
    b: numpy.ndarray
        B matrix.

    Returns
    -------
    x: numpy.ndarray
        Unknown variables.
    """
    x = numpy.dot(numpy.linalg.inv(a), b)

    return x


def gauss_eliminination(a, b):
    """Solve systems of linear equations (Ax=B) via Gauss-Elimination method.

    Parameters
    ----------
    a: numpy.ndarray
        A matrix (square matrix).
    b: numpy.ndarray
        B matrix.

    Returns
    -------
    x: numpy.ndarray
        Unknown variables.
    """
    (rows, cols) = a.shape
    # elimination phase
    for row in range(0, rows-1):  # pivot equation/row
        for i in range(row+1, rows):
            if a[i, row] != 0.0:
                factor = a[i, row]/a[row, row]
                a[i, row+1:rows] = a[i, row+1:rows] - factor*a[row, row+1:rows]
                b[i] = b[i] - factor*b[row]
    # back substitution
    for k in range(rows-1, -1, -1):
        b[k] = (b[k] - numpy.dot(a[k, k+1:rows], b[k+1:rows]))/a[k, k]

    return b
