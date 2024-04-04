import abc

import numpy
import sklearn.preprocessing

class Normalizer(abc.ABC):
    r"""A transformation on the training data of a surrogate

    Prior to training a surrogate model, a transformation can be applied
    to the data to put it on a different scale or otherwise normalize it.
    Transformed data can potentially result in a better behaved surrogate,
    depending on the normalization and the surrogate. The output of the
    surrogate trained on normalized data must be unnormalized to match the
    real function.

    This class specifies a type of normalization, defined by a
    transform function :meth:`transform` and an inverse transform
    :meth:`inverse_transform`. :meth:`transform` normalizes the training
    data for the surrogate, and :meth:`inverse_transform` unnormalizes
    the output of the surrogate. If defined correctly, applying
    :meth:`transform` and :meth:`inverse_transform` sequentially should
    return the initial data.
    ``original_data`` is the original, unnormalized training data used to
    calibrate :meth:`transform` and :meth:`inverse_transform`, if
    calibration is necessary.
    """

    def __init__(self):
        self._original_data = None

    @property
    def original_data(self):
        """numpy.ndarray: training data before normalization."""
        return self._original_data

    @original_data.setter
    def original_data(self, value):
        value = numpy.array(value, ndmin=1)
        self._original_data = value
        self._training_data = self.transform(self.original_data)

    def fit(self, x):
        """Obtain the original training data

        Method for obtaining the original training data. To be overridden 
        should a child class require the training data for calculations.

        Parameters
        ----------
        x : numerical data
            the training data

        Returns
        -------
        Normalizer
            the normalizer
        """
        self.original_data = x
        return self

    def fit_transform(self,x):
        """Preform a fit and transform the data

        Parameters
        ----------
        x : numerical data
            the original training data

        Returns
        -------
        numerical data
            transformed data
        """
        return self.fit(x).transform(x)

    @abc.abstractmethod
    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        pass

    def check_normalize(self, x):
        """Check error from normalizing process
        If defined correctly, performing :meth:`inverse_transform` on
        the output of :meth:`transform` should return the input of
        :meth:`transform`. As theory does not always align with practice,
        this method checks if the data changes from its initial
        value after the transform and inverse transform are done in
        sequence. If the data doesn't change, it returns True. If it
        does change, it returns False.

        Parameters
        ----------
        x : numerical data
            data to compare before and after transformation

        Returns
        --------
        error : float
            statistic to represent how data changes from initial value
        """
        x = numpy.array(x)
        new_x = self.inverse_transform(self.transform(x))
        return numpy.allclose(x,new_x)


class NullNormalizer(Normalizer):
    """ "Normalizer that does not modify data

    Using this normalizer will not transform the data in any way
    and is equilivant to not doing any normalization or transformation.
    """

    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        return x

    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        return x


class IntervalNormalizer(Normalizer):
    """Scales data onto the interval [0,1]

    Using the min and max of the original training data, the training
    data is normalized to the range [0,1].

    The tranformation equation is
    ..math::
        y = (x - min_val/(max_val - min_val)
        where :math:min_val is the minimum of the training data and
        :math:max_data is the maximum of the training data

    Should any subsequent data fall outside of the range established
    by the original training data, that data will outside the range [0,1]

    The inverse transformation equation is
    ..math::
        y = x * (max_val - min_val) + min_val

    In the case where the original training data contains one variable,
    and thus the min and the max are the same, no normalization is
    applied.

    The properties ``min_val`` and ``max_val`` store the min and max of
    the original training data.
    """

    def __init__(self):
        super().__init__()
        self._min_val = None
        self._max_val = None

    @property
    def min_val(self):
        """float: min of original training data"""
        if not self.original_data is None and self._min_val is None:
            self._min_val = numpy.min(self.original_data)
        return self._min_val

    @property
    def max_val(self):
        """float: max of original training data"""
        if not self.original_data is None and self._max_val is None:
            self._max_val = numpy.max(self.original_data)

        return self._max_val

    def transform(self, x):
        """Normalize the data using the min and max

        Using the min and max of the training data, the input is
        scaled such that the max of the original training data is 1 and
        the min of the original traning data is 0.

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data

        Raises
        ------
        ValueError
            the original data was never assigned
        """
        if self.original_data is None:
            raise ValueError("The original data was never given")
        x = numpy.array(x)
        if self.min_val >= self.max_val:
            return x
        else:
            return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, x):
        """Inverse normalization function

        Using the min and max of the training data, the input is
        scaled such that 1 becomes the max of the original training data
        and 0 becomes the min of the original training data

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data

        Raises
        ------
        ValueError
            the original data was never assigned
        """
        if self.original_data is None:
            raise ValueError("The original data was never given")
        x = numpy.array(x)
        if self.min_val >= self.max_val:
            return x
        else:
            return x * (self.max_val - self.min_val) + self.min_val


class ZScoreNormalizer(Normalizer):
    """Normalizes data by setting the mean to 0 and std to 1

    Using the mean and standard deviation of the original training data,
    the training data is normalized so the mean becomes 0 and the standard
    deviation becomes 1.

    The tranformation equation is
    ..math::
        y = (x - mean_data) / std_data
        where :math:mean_data is the mean of the training data and
        :math:std_data is the standard deviation of the training data

    The inverse transformation equation is
    ..math::
        y = x * std_data + mean_data

    In the case where the original training data contains one variable,
    and thus the min and the max are the same, no normalization is
    applied.

    The properties ``mean_val`` and ``std_val`` store the mean and
    standard deviation of the original training data.
    """

    def __init__(self):
        super().__init__()
        self._mean_val = None
        self._std_val = None

    @property
    def mean_val(self):
        """float: mean of original training data"""
        if not self.original_data is None and self._mean_val is None:
            self._mean_val = numpy.mean(self.original_data)
        return self._mean_val

    @property
    def std_val(self):
        """float: std of original training data"""
        if not self.original_data is None and self._std_val is None:
            try:
                self._std_val = float(numpy.std(
                    numpy.array(self.original_data, dtype=numpy.float128)
                ))
            except AttributeError:
                self._std_val = numpy.std(self.original_data)
        return self._std_val

    def transform(self, x):
        """Normalize the data using the min and max

        Using the min and max of the training data, the input is
        scaled such that the max of the original training data is 1 and
        the min of the original traning data is 0.

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data

        Raises
        ------
        ValueError
            the original data was never assigned
        """
        if self.original_data is None:
            raise ValueError("The original data was never given")
        x = numpy.array(x)
        return (x - self.mean_val) / (self.std_val)

    def inverse_transform(self, x):
        """Inverse normalization function

        Using the min and max of the training data, the input is
        scaled such that 1 becomes the max of the original training data
        and 0 becomes the min of the original training data

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data

        Raises
        ------
        ValueError
            the original data was never assigned
        """
        if self.original_data is None:
            raise ValueError("The original data was never given")
        x = numpy.array(x)
        return x * self.std_val + self.mean_val


class SymmetricalLogNormalizer(Normalizer):
    r"""Transforms data onto the symmetrical logarithm scale
    
    A logarithmic scale is a nonlinear scale that is commonly used to 
    display data that grows exponentially or has a range with many 
    orders of magnitude. A logarithmic transform can only accept values 
    greater than zero. To get around this, a more flexible version of the 
    logarithmic scale can be used, known as the symmetrical log scale. 
    This scale gets around the issue of log(0) being undefined by keeping
    the interval that contains 0 linear. In this transformation, negative
    outputs are the result of negative inputs, while in a log transformation
    negative outputs are the result of inputs less than 1. 

    The tranformation equation is 
    ..math::
        y = \begin{cases}
                \log_{10}(x/c+1) & \text{if } x > 0 \\
                -\log_{10}(-x/c+1)  & \text{if } x < 0 \\
                0 & \text{if } x = 0 \\
            \end{cases}
    
    in piece-wise form, and  
    ..math::
        y = \sgn(x) \log_{10}(1+\abs(x/c))
    
    using the sign function, where c is a constant that can be adjusted to 
    refine the linear interval near 0.

    The inverse transformation equation is
    ..math::
        y = \begin{cases}
                c(-1 + 10^{x}) & \text{if } x > 0 \\
                c(1 - 10^{-x})  & \text{if } x < 0 \\
                0 & \text{if } x = 0 \\
            \end{cases}
    
    in piece-wise form, and  
    ..math::
        y = \sgn(x) c (-1 + 10^{abs(x)})
    """

    def __init__(self, linthresh=1):
        super().__init__()
        self._linthresh = linthresh

    @property
    def linthresh(self):
        """constant to determine size of linear interval around 0"""
        return self._linthresh

    @linthresh.setter
    def linthresh(self, value):
        if value <= 0:
            raise ValueError("linthresh must be greater than 0.")
        self._linthresh = value

    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        return numpy.sign(x) * numpy.log10(1 + numpy.abs(x) / self.linthresh)

    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        return numpy.sign(x) * self.linthresh * (-1 + numpy.power(10, numpy.abs(x)))