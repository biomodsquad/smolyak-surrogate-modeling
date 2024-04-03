import pytest
import warnings

import numpy

from smolyay.normalize import (Normalizer,NullNormalizer,
                               SymmetricalLogNormalizer,IntervalNormalizer)


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

def test_check_symlog():
    """Test if the SymmetricalLogNormalizer passes the check"""
    x = [1,2,3,4,5]
    normal = SymmetricalLogNormalizer()
    assert numpy.isclose(normal.check_normalize_error(x),0)
