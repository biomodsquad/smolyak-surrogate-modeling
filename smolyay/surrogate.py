import numpy

from smolyay.grid import IndexGridGenerator


class Surrogate:
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    and sampling method, a set of grid points and their corresponding
    basis functions can be generated (``grids_generator``).
    :class:`IndexGridGenerator` contain different methods,
    such as :class:`SmolyakGridGenerator`, where a set of grid points
    and functions are generated depending on different approaches that
    make 1D points between -1 and 1 and their corresponding basis functions.
    Then, given a dimensionality, the tensor products of
    those 1D points generate multi-dimensional grid points and functions.
    Examples are :class:`SmolyakGridGenerator` which makes the grids based on
    sparse sampling and :class:`TensorGridGenerator` making them through
    full tensor product.

    ``domain`` is the domain of the real function to be approximated.
    Through linear transformation, grid points in the real
    function's domain are generated. Coefficients of the surrogate then
    will be generated by solving overdetermined systems of linear
    equations. A square matrix (number of grid points * number of grid points)
    is generated. meth:`train`, generates the coefficients of the surrogate of
    a ``real_function`` depending on the ``linear_solver`` which determines
    the method that can be used for solving linear equations (Ax=B).
    A is square matrix, x is the matrix of coefficients, and B is the matrix
    where ``real_function`` is evaluated at grid points (see example).

    :meth:`train_from_data` computes the coefficients based on
    the a set of data (function values at transformed grid points).

    Once surrogate is constructed, one can evaluate the surrogate
    through :meth:`__call__(x)` at given input.

    Parameters
    ----------
    domain: list
        Domain of the real function to be approximated.
    grids_generator: IndexGridGenerator
        Set of grids and their corresponding basis function.

    Raises
    ------
    TypeError
        Grids must be :class:`IndexGridGenerator`.

    Example
    -------
    Consider a function :math:F(y) with one independent variable.
    Basis function B_n(x) is chosen for the approximation and
    three grid points and their corresponding basis functions
    are generated in (-1, 1):
    (:math:x_0, :math:x_1, :math:x_2)
    [:math:B_0(x), :math:B_1(x) :math:B_2(x)]
    Based on the domain of the approximation, grid points of the basis
    functions can be moved to the domain, resulting in transformed grid points:
    (:math:y_0, :math:y_1, :math:y_2)
    Then:
    ..math:
        \[
        \begin{bmatrix}
        B_0(x_0)& B_1(x_0) & B_2(x_0) \\
        B_0(x_1)& B_1(x_1) & B_2(x_1) \\
        B_0(x_2)& B_1(x_2) & B_2(x_2)
        \end{bmatrix}
        \begin{bmatrix}
        C_0\\C_1\\C_2
        \end{bmatrix}
        =\begin{bmatrix}
        F(y_0)\\F(y_1)\\F(y_2)
        \end{bmatrix}
        \]

    Coefficients (x) can be obtained by solving linear equations,
    since we have a full rank system of matrixes. The user can
    select between 'lower upper decomposition' ('lu'), and
    'inverse' ('inv') method for calculating the surrogate's (S)
    coefficients.

    S = C_0B_0(x) + C_1B_1(x) + C_2B_2(x)
    """

    def __init__(self, domain, grids_generator):
        self.domain = list(domain)
        self.grids_generator = grids_generator
        self._dimension = len(list(domain))
        self._grids = self.grids_generator(self._dimension)
        self._coefficients = None

    @property
    def domain(self):
        """list: domain of the real function to be approximated."""
        return self._domain

    @domain.setter
    def domain(self, domain_real_function):
        self._domain = domain_real_function

    @property
    def grids_generator(self):
        """:class:`IndexGridGenerator`: Generator used for making the grids."""
        return self._grids_generator

    @grids_generator.setter
    def grids_generator(self, grids_generator):
        if not isinstance(grids_generator, IndexGridGenerator):
            raise TypeError('Grids must be IndexGridGenerator')
        self._grids_generator = grids_generator

    def _make_basis_matrix(self):
        """Generate basis matrix."""
        self._basis_matrix = numpy.ones((len(self._grids.points),
                                         len(self._grids.points)))
        point_num = 0
        for points in self._grids.points:
            for dimension_ in range(self._dimension):
                grid_num = 0
                for basis_function in list(zip(
                        *self._grids.basis_functions))[dimension_]:
                    self._basis_matrix[point_num][grid_num] *= (
                        basis_function(points[dimension_]))
                    grid_num += 1
            point_num += 1

    def train(self, real_function, linear_solver='lu'):
        """Fit surrogate's components (basis functions) to the function.

        Parameters
        ----------
        real_function: callabe
            Function to be approximated.
        linear_slover: string
            Method of solving linear equations:
                'lu': lower upper decomposition
                'inv': solve for coefficients through x = A^{-1}B.

        Returns
        -------
        list
            Coefficients of the surrogate.
        """
        self.real_function = real_function
        self.linear_solver = linear_solver
        self._make_basis_matrix()

        # transform grid points
        transformed_grids = numpy.zeros((
            len(self._grids.points), self._dimension))
        for dimension_ in range(self._dimension):
            transformed_grids[:, dimension_] = (
                numpy.polynomial.polyutils.mapdomain(
                    numpy.array(self._grids.points)[:, dimension_],
                    (-1, 1), self.domain[dimension_]))

        # evaluate real function at grid points
        real_function_at_grids = numpy.zeros((len(self._grids.points)))
        for point_num in range(len(self._grids.points)):
            real_function_at_grids[point_num] = (
                self.real_function(*transformed_grids[point_num]))

        # fit
        if self.linear_solver == 'lu':
            self._coefficients = numpy.linalg.solve(
                self._basis_matrix, real_function_at_grids)
        elif self.linear_solver == 'inv':
            self._coefficients = numpy.dot(
                numpy.linalg.inv(self._basis_matrix), real_function_at_grids)

        return self._coefficients.tolist()

    def train_from_data(self, data, linear_solver='lu'):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        data: list
            Real function at grid points.
        linear_slover: string
            Method of solving linear equations:
                'lu': lower upper decomposition
                'inv': solve for coefficients through x = A^{-1}B.

        Returns
        -------
        list
            Coefficients of the surrogate.

        Raises
        ------
        IndexError
            Data should be of length of the grid points.
        """
        if len(data) != len(self._grids.points):
            raise IndexError("Data must be in length of the grid points.")

        self.data = data
        self.linear_solver = linear_solver
        self._make_basis_matrix()

        # fit
        if self.linear_solver == 'lu':
            self._coefficients = numpy.linalg.solve(
                self._basis_matrix, data)
        elif self.linear_solver == 'inv':
            self._coefficients = numpy.dot(
                numpy.linalg.inv(self._basis_matrix),
                data)

        return self._coefficients

    def __call__(self, x):
        """Evaluate surrogate at a given input.

        Parameter
        ---------
        x: list (for dimension > 1), or float (for dimension = 1)
            Input.

        Returns
        -------
        float
            Surrogate output at x.

        Raises
        ------
        ValueError
            For surrogate to be evaluated, function needs to be trained.

        IndexError
            Input of the surrogate must be of length of the
            real function's dimensionality.
        """
        if self._coefficients is None:
            raise ValueError('Function needs training!')
        if len(x) == 1:
            x = list(x)
        if len(x) != self._dimension:
            raise IndexError('Input must be of length of the dimension.')

        # transform grids into basis domain
        input_surrogate = []
        for dimension_ in range(self._dimension):
            basis_domain_grid = numpy.polynomial.polyutils.mapdomain(
                x[dimension_], self.domain[dimension_], (-1, 1))
            input_surrogate.append(basis_domain_grid)

        # evaluate surrogate
        surrogate_output = 0
        for index_point in range(len(self._grids.points)):
            output = 1
            for dimension_ in range(self._dimension):
                output *= (self._grids.basis_functions[index_point]
                           [dimension_](input_surrogate[dimension_]))
            output *= self._coefficients[index_point]
            surrogate_output += output

        return surrogate_output
