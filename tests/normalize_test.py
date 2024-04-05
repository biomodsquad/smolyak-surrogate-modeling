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


@pytest.mark.parametrize(
    "normal",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_fit(normal):
    """Test that fit returns Normalizer and sets original_data"""
    x = [1, 2, 3, 4, 5]
    n = normal.fit(x)
    assert isinstance(n, Normalizer)
    assert n == normal
    assert numpy.array_equal(n.original_data, [1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "normal",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_fit_transform(normal):
    """Test that original_data is set using fit_transform and not transform"""
    x = [1, 2, 3, 4, 5]
    x_reverse = [5, 4, 3, 2, 1]
    y = normal.fit_transform(x)
    after_fit = normal.original_data
    y_reverse = normal.transform(x_reverse)
    assert numpy.array_equal(after_fit, [1, 2, 3, 4, 5])
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert numpy.array_equal(y, numpy.flip(y_reverse))
    assert not numpy.array_equal(y, y_reverse)


@pytest.mark.parametrize(
    "normal",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_fit_transform_inverse(normal):
    """Test original_data is not set using inverse_transform"""
    x = [1, 2, 3, 4, 5]
    x_reverse = [5, 4, 3, 2, 1]
    normal.fit_transform(x)
    after_fit = normal.original_data
    normal.inverse_transform(x_reverse)
    assert numpy.array_equal(after_fit, [1, 2, 3, 4, 5])
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "normal",
    [
        NullNormalizer(),
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
    ],
)
def test_check(normal):
    """Test if the all the normalizers pass the check"""
    x = [1, 2, 3, 4, 5]
    x2 = numpy.array(numpy.linspace(-10, 10), ndmin=2).transpose()
    x3 = numpy.array(
        [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 2, 5, 9, 0], [1, 2, 3, 4, 5]]
    )
    normal.fit(x)
    assert normal.check_normalize(x)
    assert normal.check_normalize(x2)
    assert normal.check_normalize(x3)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])


def test_null_attributes():
    """Test if the attributes of NullNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])


def test_null_transform():
    """Test that the NullNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), [1, 2, 3, 4, 5])


def test_null_inverse():
    """Test that the NullNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = NullNormalizer()
    assert numpy.allclose(normal.inverse_transform(x), [1, 2, 3, 4, 5])


def test_interval_attributes():
    """Test if the attributes of IntervalNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert normal.max_val == 5
    assert normal.min_val == 1


def test_interval_transform():
    """Test that the IntervalNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), [0, 0.25, 0.5, 0.75, 1])


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


def test_interval_refit():
    """Test that min and max are recaluated if original_data is changed"""
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3, 4, 5, 6]
    min2 = numpy.min(x2)
    max2 = numpy.max(x2)
    normal = IntervalNormalizer()
    normal.fit(x1)
    normal.fit(x2)
    assert numpy.array_equal(normal.original_data, x2)
    assert normal.min_val == min2
    assert normal.max_val == max2


def test_symlog_attributes():
    """Test if the attributes of SymmetricalLogNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert normal.linthresh == 1


def test_symlog_transform():
    """Test that the SymmetricalLogNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), numpy.log10([2, 3, 4, 5, 6]))


def test_symlog_inverse():
    """Test that the SymmetricalLogNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = SymmetricalLogNormalizer()
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), [9, 99, 999, 9999, 99999])


def test_zscore_attributes():
    """Test if the attributes of ZScoreNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert normal.mean_val == 3
    assert normal.std_val == numpy.sqrt(2)


def test_zscore_transform():
    """Test that the ZScoreNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert numpy.allclose(
        normal.transform(x),
        [
            -2 / numpy.sqrt(2),
            -1 / numpy.sqrt(2),
            0,
            1 / numpy.sqrt(2),
            2 / numpy.sqrt(2),
        ],
    )


def test_zscore_inverse():
    """Test that the ZScoreNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert numpy.allclose(
        normal.inverse_transform(x),
        [
            numpy.sqrt(2) + 3,
            2 * numpy.sqrt(2) + 3,
            3 * numpy.sqrt(2) + 3,
            4 * numpy.sqrt(2) + 3,
            5 * numpy.sqrt(2) + 3,
        ],
    )


def test_zscore_transform_error():
    """Test if ZScoreNormalizer returns an error without original data"""
    normal = ZScoreNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1, 2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1, 2])
    with pytest.raises(ValueError):
        normal.check_normalize([1, 3])


def test_zscore_refit():
    """Test that mean and std are recaluated if original_data is changed"""
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3, 4, 5, 6]
    mean2 = numpy.mean(x2)
    std2 = numpy.std(x2)
    normal = ZScoreNormalizer()
    normal.fit(x1)
    normal.fit(x2)
    assert numpy.array_equal(normal.original_data, x2)
    assert normal.mean_val == mean2
    assert normal.std_val == std2


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
    """Test if SklearnNormalizer passes the check for multiple transformers"""
    x = numpy.array([1, 2, 3, 4, 5])
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
    x = numpy.array(
        [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 2, 5, 9, 0], [1, 2, 3, 4, 5]]
    )
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    y = normal.transform(x)
    new_x = normal.inverse_transform(y)
    assert normal.check_normalize(x)
    assert y.shape == x.shape
    assert y.shape == new_x.shape
    assert numpy.allclose(x, new_x)


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
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert isinstance(normal.scalar, scalar_class)


def test_sklearn_transform():
    """Test that the SklearnNormalizer transform is correct"""
    x = [1, 2, 3, 4, 5]
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), [0.2, 0.4, 0.6, 0.8, 1])


def test_sklearn_transform_multidim():
    """Test if transform returns correct answer for multidimensional data"""
    x = numpy.array(
        [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 2, 5, 9, 10], [1, 2, 3, 4, 5]]
    )
    true_y = x / 10
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    normal_y = normal.transform(x)
    assert normal_y.shape == true_y.shape
    assert numpy.array_equal(normal_y, true_y)


def test_sklearn_inverse():
    """Test that the SklearnNormalizer inverse transform is correct"""
    x = [1, 2, 3, 4, 5]
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), [5, 10, 15, 20, 25])


def test_sklearn_inverse_multidim():
    """Test inverse_transform for multidimensional data"""
    x = numpy.array(
        [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 2, 5, 9, 10], [1, 2, 3, 4, 5]]
    )
    true_y = x * 10
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    normal_y = normal.inverse_transform(x)
    assert normal_y.shape == true_y.shape
    assert numpy.array_equal(normal_y, true_y)


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
def test_sklearn_needs_fit(scalar_class):
    """Test that error is returned if the scalar is not fit"""
    x = [1, 2, 3, 4, 5]
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    with pytest.raises(ValueError):
        normal.transform(x)


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
def test_sklearn_fit(scalar_class):
    """Test that original_data is set using fit"""
    x = numpy.array(numpy.linspace(-10, 10), ndmin=2).transpose()
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.array_equal(normal.original_data, x)
    try:
        normal.scalar.transform(x)
    except ValueError as exc:
        assert False, f"transform raised an exception {exc}"


def test_sklearn_refit():
    """Test that scalar is refitted if original_data is changed"""
    x = [1, 2, 3, 4, 5]
    y = [0.2, 0.4, 0.6, 0.8, 1]
    x2 = [1, 2, 3, 4, 5, 6]
    y2 = numpy.divide(x, 6)
    scalar = sklearn.preprocessing.MaxAbsScaler()
    normal = SklearnNormalizer(scalar)
    normal.fit(x)
    assert numpy.array_equal(normal.transform(x), y)
    normal.fit(x2)
    assert numpy.array_equal(normal.original_data, x2)
    print(normal.transform(x))
    assert numpy.array_equal(normal.transform(x), y2)


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
def test_sklearn_fit_transform(scalar_class):
    """Test that original_data is set using fit_transform and not transform"""
    x = [1, 2, 3, 4, 5]
    x_reverse = [5, 4, 3, 2, 1]
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    y = normal.fit_transform(x)
    y_reverse = normal.transform(x_reverse)
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
    assert numpy.array_equal(y, numpy.flip(y_reverse))


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
def test_sklearn_fit_transform_inverse(scalar_class):
    """Test original_data is not set using inverse_transform"""
    x = [1, 2, 3, 4, 5]
    x_reverse = [5, 4, 3, 2, 1]
    scalar = scalar_class()
    normal = SklearnNormalizer(scalar)
    normal.fit_transform(x)
    after_fit = normal.original_data
    normal.inverse_transform(x_reverse)
    assert numpy.array_equal(after_fit, [1, 2, 3, 4, 5])
    assert numpy.array_equal(normal.original_data, [1, 2, 3, 4, 5])
