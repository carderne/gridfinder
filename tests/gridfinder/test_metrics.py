"""
Test suite for the metrics module
"""
import os
from typing import Optional

import pytest

import rasterio
from affine import Affine
import numpy as np
import geopandas as gp
from shapely.geometry import LineString
from rasterio.enums import Resampling
import rasterio.warp


TRANSFORM = Affine(1 + 1e-10, 0.0, 0.0, 0.0, 1 + 1e-10, 0.0)


def accuracy(ground_truth_lines: gp.GeoDataFrame,
             raster_guess: rasterio.DatasetReader, cell_size_in_meters: Optional[int]):
    """
    Calculates the accuracy of a grid line prediction based the provided ground truth.
    The cell_size_in_meters parameter controls the size of one rectangular area
    in the prediction which is either marked as grid line (1) or no grid line (0).




    """
    def get_scaling_factor():
        pass

    def perform_scaling():
        pass

    def rasterize_ground_truth():
        pass

    def measure_cell_wise_accuracy():
        pass

    return 1.0


def store_tif_file(file_path, raster: np.array) -> str:
    with rasterio.open(
        str(file_path),
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs="EPSG:3857",
        transform=TRANSFORM,
    ) as dst:
        dst.write(raster, 1)
        dst.close()
    return str(file_path)


CORRECT_GUESS = np.array(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     ], dtype=np.float32
)


CORRECT_GUESS_BUT_SHIFTED_LEFT = np.array(
    [[0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     ], dtype=np.float32
)

PARTIALLY_CORRECT_GUESS = np.array(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     ], dtype=np.float32
)

PARTIALLY_CORRECT_GUESS_BUT_SHIFTED_LEFT = np.array(
    [[0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     ], dtype=np.float32
)


# The value don't map the indices in the grid directly
# The later used rasterize function start counting the indices form one instead of zero
# Also the LineString coordinates need to be transposed
GROUND_TRUTH = gp.GeoDataFrame(
    index=[0],
    geometry=[LineString([[2, 2], [2, 3], [2, 4], [2, 5]])],
    crs="EPSG:3857"
)


@pytest.fixture()
def ground_truth_lines():
    return GROUND_TRUTH


@pytest.fixture()
def correct_guess(raster_saver) -> str:
    return raster_saver(CORRECT_GUESS, "correct_guess.tif")


@pytest.fixture()
def correct_guess_but_shifted_left(raster_saver) -> str:
    return raster_saver(CORRECT_GUESS_BUT_SHIFTED_LEFT, "correct_guess_but_shifted_left.tif")


@pytest.fixture()
def partially_correct_guess(raster_saver) -> str:
    return raster_saver(PARTIALLY_CORRECT_GUESS, "partially_correct_guess.tif")


@pytest.fixture()
def partially_correct_guess_but_shifted_left(raster_saver) -> str:
    return raster_saver(PARTIALLY_CORRECT_GUESS_BUT_SHIFTED_LEFT, "partially_correct_guess_but_shifted_left.tif")


@pytest.fixture()
def raster_saver(tmpdir_factory):
    """ Saves Raster as TIF file"""
    def _raster_saver(raster: np.array, file_name: str):
        tmp_folder = tmpdir_factory.mktemp("tmp_raster")
        path = os.path.join(tmp_folder, file_name)
        store_tif_file(path, raster)
        return rasterio.open(path)
    return _raster_saver


@pytest.mark.parametrize(
    ["raster_guess", "cell_size_in_meters", "expected_accuracy"],
    [
        (pytest.lazy_fixture("correct_guess"), None, 1.0),
        (pytest.lazy_fixture("correct_guess_but_shifted_left"), None, 0.0),
        (pytest.lazy_fixture("partially_correct_guess"), None, 0.75),
        (pytest.lazy_fixture("partially_correct_guess_but_shifted_left"), None, 0.0),
        (pytest.lazy_fixture("correct_guess"), 2, 1.0),
        (pytest.lazy_fixture("correct_guess_but_shifted_left"), 2, 1.0),
        (pytest.lazy_fixture("partially_correct_guess"), 2, 0.75),
        (pytest.lazy_fixture("partially_correct_guess_but_shifted_left"), 2, 0.75),
    ]
)
def test_accuracy(ground_truth_lines: gp.GeoDataFrame,
                  raster_guess: rasterio.DatasetReader,
                  cell_size_in_meters: Optional[int],
                  expected_accuracy: float):
    assert pytest.approx(accuracy(
        ground_truth_lines,
        raster_guess,
        cell_size_in_meters=cell_size_in_meters
    ), 0.01) == expected_accuracy

