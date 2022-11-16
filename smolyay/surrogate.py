import numpy

from smolyay.grid import IndexGridGenerator

class Surrogate:
    r"""Create a surrogate to approximate a complex function.
    Depending on the dimensionality (number of independent variables),
    and sampling method, a set of grid points and their corresponding
    basis functions can be generated (``grid_generator``).
    :class:`IndexGridGenerator` contain different methods,
    such as :class:`SmolyakGridGenerator`, where a set of grid points
    and functions are generated depending on different approaches that
    make 1D points between -1 and 1 and their corresponding basis functions.
    Then, given a dimensionality, the tensor products of
    those 1D points generate multi-dimensional grid points and functions.
    Examples are :class:`SmolyakGridGenerator` which makes the grids based on
    sparse sampling and :class:`TensorGridGenerator` making them through
    full tensor product.
    ``domain`` is the domain of the function to be approximated.
    Through linear transformation, grid points in the
    function's domain are generated (:meth:`_mapdomain`).
    ``coefficients`` of the surrogate then will be generated by solving
    overdetermined systems of linear equations.
    A square matrix (number of grid points * number of grid points)
    is generated by evaluating basis functions at grid points.
    :meth:`train`, generates the coefficients of the surrogate of
    a ``function`` depending on the ``linear_solver`` which determines
    the method that can be used for solving linear equations (Ax=B).
    A is square matrix, x is the matrix of coefficients, and B is the matrix
    where ``function`` is evaluated at transformed grid points (see example).
    ``points`` stores the points that are used for sampling.
    :meth:`train_from_data` computes the coefficients based on
    the a set of data (function values at transformed grid points).
    Once surrogate is constructed, one can evaluate the surrogate
    through :meth:`__call__(x)` at given input.
    :meth:`_reset_surrogate` resets the surrogate's coefficients,
    sampled points, original ``grids``(holds the grid points
    and their corresponding basis fucntions depending on the
    dimensionality and the generator), and ``data`` (function at
    sampling points).

    Parameters
    ----------
    domain: list
        Domain of the function to be approximated.
    grid_generator: IndexGridGenerator
        Set of grids and their corresponding basis function.

    Raises
    ------
    TypeError
        Grids must be :class:`IndexGridGenerator`.
        Domain must be dim x 2.

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
    coefficients. 'lstsq' solves the above equations through
    least linear method.
    S = C_0B_0(x) + C_1B_1(x) + C_2B_2(x)
    """

    def __init__(self, domain, grid_generator):
        self.domain = domain
        self.grid_generator = grid_generator
        self._coefficients = None
        self._grid = None
        self._points = None
        self._data = None

    @property
    def coefficients(self):
        """list: coefficients of surrogate."""
        if self._coefficients is not None:
            return self._coefficients.tolist()
        else:
            return None

    @property
    def data(self):
        """list: data at sampling grid points."""
        if self._data is not None:
            return self._data.tolist()
        else:
            return None

    @property
    def dimension(self):
        """Numbers of independent variables."""
        return self._domain.shape[0]

    @property
    def domain(self):
        """list: domain of the function to be approximated."""
        if self.dimension > 1:
            return self._domain.tolist()
        else:
            return self._domain[0].tolist()

    @domain.setter
    def domain(self, value):
        value = numpy.array(value, ndmin=2)
        if value.ndim != 2 or value.shape[1] != 2:
            raise TypeError('Domain must be dim x 2')
        self._domain = value
        self._reset_grid()

    @property
    def grid(self):
        """:class:`IndexGrid` Sampling grid."""
        if self._grid is None:
            self._grid = self.grid_generator(self.dimension)
        return self._grid

    @property
    def grid_generator(self):
        """:class:`IndexGridGenerator`: Sampling grid generator."""
        return self._grid_generator

    @grid_generator.setter
    def grid_generator(self, value):
        if not isinstance(value, IndexGridGenerator):
            raise TypeError('Grids must be IndexGridGenerator')
        self._grid_generator = value
        self._reset_grid()

    @property
    def points(self):
        """numpy.ndarray: Points to be sampled."""
        if self._points is None:
            self._points = self._mapdomain(
                self.grid.points, [[-1, 1]]*self.dimension, self._domain)
        return self._points

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
            function's dimensionality.
        """
        if self._coefficients is None:
            raise ValueError('Function needs training!')
        x = numpy.array(x, copy=False, ndmin=1)
        if x.shape != (self.dimension, ):
            raise IndexError('Input must match dimension of domain.')

        # transform point into basis domain and evaluate
        x_scaled = self._mapdomain(x, self._domain, [[-1, 1]]*self.dimension)
        value = 0
        for coeff, basis in zip(self._coefficients, self.grid.basis_functions):
            if self.dimension > 1:
                term = numpy.product(
                    [basis_i(x_i) for basis_i, x_i in zip(basis, x_scaled)])
            else:
                term = basis(x_scaled[0])
            value += coeff*term

        return value

    def train(self, function, linear_solver='lu'):
        """Fit surrogate's components (basis functions) to the function.

        Parameters
        ----------
        function: callabe
            Function to be approximated.
        linear_slover: string
            Method of solving linear equations:
                'lu': lower upper decomposition.
                'inv': solve for coefficients through x = A^{-1}B.
                'lstsq': least-square solution.
        """
        self._data = [function(point) for point in self.points]
        self.train_from_data(self._data, linear_solver)

    def train_from_data(self, data, linear_solver='lu'):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        data: list
            function at grid points.
        linear_slover: string
            Method of solving linear equations:
                'lu': lower upper decomposition
                'inv': solve for coefficients through x = A^{-1}B.
                'lstsq': least-square solution.

        Raises
        ------
        IndexError
            Data should be of length of the grid points.
        ValueError
            Solver should be selected between defined methods.
        """
        self._data = data
        self._data = numpy.array(self._data, copy=False, ndmin=1)
        if self._data.shape != (numpy.array(self.grid.points).shape[0],):
            raise IndexError("Data must be same length as grid points.")

        # make basis matrix
        points, basis_functions = self.grid.points, self.grid.basis_functions
        basis_matrix = numpy.zeros((len(points), len(points)))
        for i, point in enumerate(points):
            for j, basis in enumerate(basis_functions):
                if self.dimension > 1:
                    value = numpy.prod([f(x) for x, f in zip(point, basis)])
                else:
                    value = basis(point)
                basis_matrix[i, j] = value

        if linear_solver == 'lu':
            self._coefficients = numpy.linalg.solve(
                basis_matrix,
                self._data)
        elif linear_solver == 'inv':
            self._coefficients = numpy.dot(numpy.linalg.inv(
                basis_matrix),
                self._data)
        elif linear_solver == 'lstsq':
            self._coefficients = numpy.linalg.lstsq(basis_matrix,
                                                    self._data,
                                                    rcond=None)[0]
        else:
            raise ValueError('Solver not recognized')

    @staticmethod
    def _mapdomain(x, old, new):
        """Transform the points into new domain.

        Parameters
        ----------
        x: list
            point(s) to be transformed.
        old: list
            old domain.
        new: list
            new domain.

        Returns
        -------
        numpy.ndarray
            Transformed point(s).

        Raises
        ------
        TypeError
            Old and new domain must have the same shape.
            Domain should be a dim x 2 array.
            Points must be an N x dim array matching domain.
        """
        old = numpy.array(old, copy=False, ndmin=2)
        new = numpy.array(new, copy=False, ndmin=2)

        # error checking
        if old.shape != new.shape:
            raise TypeError('Old and new domain must have the same shape')
        if old.ndim != 2 or old.shape[1] != 2:
            raise TypeError('Domain should be a dim x 2 array')

        return new[:,0]+(new[:,1]-new[:,0])*((x-old[:,0])/(old[:,1]-old[:,0]))

    def _reset_grid(self):
        """Reset the grids, coefficients and points."""
        self._grid = None
        self._coefficients = None
        self._points = None
        self._data = None
