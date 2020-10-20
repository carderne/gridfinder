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
from rasterio.features import rasterize


TRANSFORM = Affine(1 + 1e-10, 0.0, 0.0, 0.0, 1 + 1e-10, 0.0)


def get_resolution_in_meters(reader: rasterio.io.DatasetReader) -> tuple:
    if reader.crs.to_string() != "EPSG:3857":
        # get the resolution in meters via cartesian crs
        transform, _, _ = rasterio.warp.calculate_default_transform(
           reader.crs.to_string(), "EPSG:3857", reader.width, reader.height, *reader.bounds
        )
        pixel_size_x = transform[0]
        pixel_size_y = -transform[4]
    else:
        pixel_size_x, pixel_size_y = reader.res

    return pixel_size_x, -pixel_size_y


def accuracy(ground_truth_lines: gp.GeoDataFrame,
             raster_guess: rasterio.DatasetReader, cell_size_in_meters: Optional[float]):
    """
    Calculates the accuracy of a grid line prediction based the provided ground truth.
    The cell_size_in_meters parameter controls the size of one rectangular area
    in the prediction which is either marked as grid line (1) or no grid line (0).




    """
    # TODO: Check buffering the ground_truth
    def get_scaling_factor(desired_cell_size: float, current_cell_size: float) -> float:
        return desired_cell_size / current_cell_size

    def perform_scaling(raster_reader: rasterio.DatasetReader, scaling_factor_x: float, scaling_factor_y: float) -> np.array:
        # TODO: check if avoiding the additional loading is possible
        scaled_raster = raster_reader.read(
            out_shape=(
                raster_reader.count,
                int(raster_reader.height * scaling_factor_x),
                int(raster_reader.width * scaling_factor_y)
            ),
            resampling=Resampling.max  # TODO: check this
        )
        return scaled_raster

    def rasterize_geo_dataframe(raster_reader: rasterio.DatasetReader, data_frame: gp.GeoDataFrame) -> np.array:
        """ All raster values where shapes are found will have the values one, the rest zero."""
        data_rows = [row.geometry for _, row in data_frame.iterrows()]
        new_raster = rasterize(
            data_rows,
            out_shape=raster_reader.shape,
            fill=0,
            default_value=1,
            all_touched=False,
            transform=raster_reader.transform,
        )
        return new_raster

    def measure_cell_wise_accuracy():
        pass


    if cell_size_in_meters is not None:
        current_cell_size_x, current_cell_size_y = get_resolution_in_meters(raster_guess)
        scaling_x = get_scaling_factor(cell_size_in_meters, current_cell_size_x)
        scaling_y = get_scaling_factor(cell_size_in_meters, current_cell_size_y)

        raster = perform_scaling(raster_guess, scaling_x, scaling_y)
    else:
        raster = raster_guess.read(1), raster_guess.transform

    raster_ground_truth = rasterize_geo_dataframe(raster_guess, ground_truth_lines)
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
        (pytest.lazy_fixture("correct_guess"), 2, 1.0),
        # (pytest.lazy_fixture("correct_guess_but_shifted_left"), None, 0.0),
        # (pytest.lazy_fixture("partially_correct_guess"), None, 0.75),
        # (pytest.lazy_fixture("partially_correct_guess_but_shifted_left"), None, 0.0),
        # (pytest.lazy_fixture("correct_guess"), 2., 1.0),
        # (pytest.lazy_fixture("correct_guess_but_shifted_left"), 2., 1.0),
        # (pytest.lazy_fixture("partially_correct_guess"), 2., 0.75),
        # (pytest.lazy_fixture("partially_correct_guess_but_shifted_left"), 2., 0.75),
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

