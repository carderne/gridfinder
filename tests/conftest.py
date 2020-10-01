import numpy as np
import pytest
from affine import Affine


@pytest.fixture()
def sample_transform():
    return Affine(13, 12, 10, 8, 2, 1)


@pytest.fixture()
def two_dim_array():
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=float)


@pytest.fixture()
def three_dim_array(two_dim_array):
    return np.array([two_dim_array, two_dim_array])
