import pytest
import warnings

import numpy

from smolyay.normalize import (
    Normalizer,
    NullNormalizer,
    SymmetricalLogNormalizer,
    IntervalNormalizer,
    ZScoreNormalizer,
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
    normal.original_data = x
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
