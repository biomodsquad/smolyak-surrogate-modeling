import pytest
import warnings

import numpy

from smolyay.normalize import (Normalizer,NullNormalizer,
                               SymmetricalLogNormalizer,IntervalNormalizer,
                               ZScoreNormalizer)


def test_check_null():
    """Test if the NullNormalizer passes the check"""
    x = [1,2,3,4,5]
    normal = NullNormalizer()
    assert normal.check_normalize_error(x) == 0

def test_check_interval():
    """Test if the IntervalNormalizer passes the check"""
    x = [1,2,3,4,5]
    normal = IntervalNormalizer()
    normal.original_data = x
    assert numpy.isclose(normal.check_normalize_error(x),0)

def test_interval_max():
    """Test if the IntervalNormalizer returns the max"""
    x = [1,2,3,4,5]
    normal = IntervalNormalizer()
    normal.original_data = x
    assert normal.max_val == 5

def test_interval_min():
    """Test if the IntervalNormalizer returns the min"""
    x = [1,2,3,4,5]
    normal = IntervalNormalizer()
    normal.original_data = x
    assert normal.min_val == 1

def test_interval_transform_error():
    """Test if IntervalNormalizer returns an error without original data"""
    normal = IntervalNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1,2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1,2]) 
    with pytest.raises(ValueError):
        normal.max_val
    with pytest.raises(ValueError):
        normal.min_val

def test_check_symlog():
    """Test if the SymmetricalLogNormalizer passes the check"""
    x = [1,2,3,4,5]
    normal = SymmetricalLogNormalizer()
    assert numpy.isclose(normal.check_normalize_error(x),0)

def test_check_zscore():
    """Test if the ZScoreNormalizer passes the check"""
    x = [1,2,3,4,5]
    normal = ZScoreNormalizer()
    normal.original_data = x
    assert numpy.isclose(normal.check_normalize_error(x),0)

def test_zscore_mean():
    """Test if the ZScoreNormalizer returns the mean"""
    x = [1,2,3,4,5]
    normal = ZScoreNormalizer()
    normal.original_data = x
    assert normal.mean_val == 3

def test_zscore_std():
    """Test if the ZScoreNormalizer returns the standard deviation"""
    x = [1,2,3,4,5]
    normal = ZScoreNormalizer()
    normal.original_data = x
    assert normal.std_val == numpy.sqrt(2)

def test_zscore_transform_error():
    """Test if ZScoreNormalizer returns an error without original data"""
    normal = ZScoreNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1,2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1,2]) 
    with pytest.raises(ValueError):
        normal.mean_val
    with pytest.raises(ValueError):
        normal.std_val