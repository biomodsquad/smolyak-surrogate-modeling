import pytest
import warnings

import numpy
import sklearn.preprocessing

from smolyay.normalize import (
    Normalizer,
    NullNormalizer,
    SymmetricalLogNormalizer,
    IntervalNormalizer,
    ZScoreNormalizer,
    SklearnNormalizer,
)


def test_check_null():
    """Test if the NullNormalizer passes the check"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    assert normal.check_normalize(x)

def test_null_attributes():
    """Test if the attributes of NullNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data,[1, 2, 3, 4, 5])

def test_null_fit():
    """Test that the NullNormalizer is fitted"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    assert isinstance(normal.fit(x),NullNormalizer)

def test_null_transform():
    """Test that the NullNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    assert numpy.allclose(normal.fit_transform(x), [1, 2, 3, 4, 5])

def test_null_inverse():
    """Test that the NullNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    assert numpy.allclose(normal.inverse_transform(x), [1, 2, 3, 4, 5])

def test_interval_check():
    """Test if the IntervalNormalizer passes the check"""
    x = numpy.array([1, 2, 3, 4, 5],ndmin=2).transpose()
    normal = IntervalNormalizer()
    normal.fit(x)
    assert normal.check_normalize(x)

def test_interval_attributes():
    """Test if the attributes of IntervalNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data,[1, 2, 3, 4, 5])
    assert normal.max_val == 5
    assert normal.min_val == 1

def test_interval_transform():
    """Test that the IntervalNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    assert numpy.allclose(normal.fit_transform(x), [0, 0.25, 0.5, 0.75, 1])

def test_interval_inverse():
    """Test that the IntervalNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), [5, 9, 13, 17, 21])

def test_interval_transform_error():
    """Test if IntervalNormalizer returns an error without original data"""
    normal = IntervalNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1, 2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1, 2])
    with pytest.raises(ValueError):
        normal.check_normalize([1, 3])

def test_symlog_check():
    """Test if the SymmetricalLogNormalizer passes the check"""
    x = numpy.array(numpy.linspace(-10,10),ndmin=2).transpose()
    normal = SymmetricalLogNormalizer()
    assert normal.check_normalize(x)

def test_symlog_attributes():
    """Test if the attributes of SymmetricalLogNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data,[1, 2, 3, 4, 5])
    assert normal.linthresh == 1

def test_symlog_transform():
    """Test that the SymmetricalLogNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.fit_transform(x), numpy.log10([2, 3, 4, 5, 6]))

def test_symlog_inverse():
    """Test that the SymmetricalLogNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), 
            [9, 99, 999, 9999, 99999])


def test_zscore_check():
    """Test if the ZScoreNormalizer passes the check"""
    x = numpy.array([1, 2, 3, 4, 5],ndmin=2).transpose()
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert normal.check_normalize(x)

def test_zscore_attributes():
    """Test if the attributes of ZScoreNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data,[1, 2, 3, 4, 5])
    assert normal.mean_val == 3
    assert normal.std_val == numpy.sqrt(2)

def test_zscore_transform():
    """Test that the ZScoreNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    assert numpy.allclose(normal.fit_transform(x), 
            [-2/numpy.sqrt(2), -1/numpy.sqrt(2), 0, 1/numpy.sqrt(2), 
                2/numpy.sqrt(2)])

def test_zscore_inverse():
    """Test that the ZScoreNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), 
            [numpy.sqrt(2) + 3,2*numpy.sqrt(2)+3, 3*numpy.sqrt(2)+3, 
                4*numpy.sqrt(2)+3, 5*numpy.sqrt(2)+3])

def test_zscore_transform_error():
    """Test if ZScoreNormalizer returns an error without original data"""
    normal = ZScoreNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1, 2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1, 2])
    with pytest.raises(ValueError):
        normal.check_normalize([1, 3])

@pytest.mark.parametrize(
    "scalar_class",
    [
        sklearn.preprocessing.StandardScaler,
        sklearn.preprocessing.MaxAbsScaler,
        sklearn.preprocessing.MinMaxScaler,
        sklearn.preprocessing.PowerTransformer,
        sklearn.preprocessing.RobustScaler,
    ],
)
def test_sklearn_check(scalar_class):
    """Test if SklearnNormalizer functions correctly"""
    x = numpy.array([1,2,3,4,5])
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert normal.check_normalize(x)

@pytest.mark.parametrize(
    "scalar_class",
    [
        sklearn.preprocessing.StandardScaler,
        sklearn.preprocessing.MaxAbsScaler,
        sklearn.preprocessing.MinMaxScaler,
        sklearn.preprocessing.PowerTransformer,
        sklearn.preprocessing.RobustScaler,
    ],
)
def test_sklearn_check_multidim(scalar_class):
    """Test if SklearnNormalizer functions correctly"""
    x = numpy.array([[1,2,3,4,5],[3,4,5,6,7],[5,2,5,9,0],[1,2,3,4,5]])
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    y = normal.transform(x)
    new_x = normal.inverse_transform(y)
    assert normal.check_normalize(x)
    assert y.shape == x.shape
    assert y.shape == new_x.shape
    assert numpy.allclose(x,new_x)

@pytest.mark.parametrize(
    "scalar_class",
    [
        sklearn.preprocessing.StandardScaler,
        sklearn.preprocessing.MaxAbsScaler,
        sklearn.preprocessing.MinMaxScaler,
        sklearn.preprocessing.PowerTransformer,
        sklearn.preprocessing.RobustScaler,
    ],
)
def test_sklearn_attributes(scalar_class):
    """Test if the attributes of SklearnNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.array_equal(normal.original_data,[1, 2, 3, 4, 5])
    assert isinstance(normal.scalar,scalar_class)

def test_sklearn_transform():
    """Test that the SklearnNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    assert numpy.allclose(normal.fit_transform(x), [0.2,0.4,0.6,0.8,1])

def test_sklearn_transform_multidim():
    """Test if SklearnNormalizer functions correctly"""
    x = numpy.array([[1,2,3,4,5],[3,4,5,6,7],[5,2,5,9,10],[1,2,3,4,5]])
    true_y = x/10
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    normal_y = normal.transform(x)
    assert normal_y.shape == true_y.shape
    assert numpy.array_equal(normal_y,true_y)

def test_sklearn_inverse():
    """Test that the SklearnNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), [5, 10, 15, 20, 25])

def test_sklearn_inverse_multidim():
    """Test if SklearnNormalizer functions correctly"""
    x = numpy.array([[1,2,3,4,5],[3,4,5,6,7],[5,2,5,9,10],[1,2,3,4,5]])
    true_y = x*10
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    normal_y = normal.inverse_transform(x)
    assert normal_y.shape == true_y.shape
    assert numpy.array_equal(normal_y,true_y)