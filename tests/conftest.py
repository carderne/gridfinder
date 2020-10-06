import os
import sys

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio import DatasetReader

sys.path.append(os.path.abspath("."))
from config import top_level_directory


@pytest.fixture(scope="session")
def test_resources():
    return os.path.join(top_level_directory, "tests", "resources")


@pytest.fixture()
def sample_transform():
    return Affine(13, 12, 10, 8, 2, 1)


@pytest.fixture()
def two_dim_array():
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=float)


@pytest.fixture()
def three_dim_array(two_dim_array):
    return np.array([two_dim_array, two_dim_array])


@pytest.fixture()
def sample_ntl_raster(test_resources) -> DatasetReader:
    with rasterio.open(os.path.join(test_resources, "sample_ntl.tif")) as dataset:
        yield dataset
