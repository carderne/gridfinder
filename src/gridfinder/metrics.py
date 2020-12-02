""" Metrics module implements calculation of confusion matrix given a prediction and ground truth. """
from typing import Optional, Tuple, List, Callable
from deprecated import deprecated

import fiona

from affine import Affine
import numpy as np
import geopandas as gp
from sklearn.metrics import confusion_matrix

import rasterio
from rasterio.enums import Resampling
import rasterio.warp
from rasterio.warp import reproject
from rasterio.features import rasterize

from gridfinder._util import clip_line_poly
from gridfinder.util.raster import get_clipped_data, get_resolution_in_meters


@deprecated(
    reason="Function is deprecated and will be removed in the next release 1.3.0"
    "Use get_binary_arrays to compute y_pred and y_true from the grid finder output.",
)
def eval_metrics(
    ground_truth_lines: gp.GeoDataFrame,
    raster_guess_reader: rasterio.DatasetReader,
    cell_size_in_meters: Optional[float] = None,
    aoi: Optional[gp.GeoDataFrame] = None,
    metrics: List[Callable] = [confusion_matrix],
) -> dict:
    """
    Calculates sklearn metrics
    of a grid line prediction based the provided ground truth.

    :param ground_truth_lines: A gp.GeoDataFrame object which contains LineString objects as shapes
                               representing the grid lines.
    :param raster_guess_reader: A rasterio.DatasetReader object which contains the raster of predicted grid lines.
                                Pixel values marked with 1 are considered a prediction of a grid line.
    :param cell_size_in_meters: The cell_size_in_meters parameter controls the size of one prediction in meters.
                                E.g. the original raster has a pixel size of 100m x 100m.
                                A cell_size of 1000m meters means that one prediction
                                is now the grouping of 100 original pixels.
                                This is done for both the ground truth raster and the prediction raster.
                                The down-sampling strategy considers a collection of pixel values as a positive
                                prediction (value = 1) if at least one pixel in that collection has the value 1.
    :param aoi: A gp.GeoDataFrame containing exactly one Polygon or Multipolygon marking the area of interest.
                The CRS is expected to be the same as the raster_guess_readers' CRS.
    :param metrics: A sklearn.metrics object describing a metric that should be evaluated

    :returns: dictionary of metrics resulting from evaluation
    """
    y_pred, y_ture = get_binary_arrays(
        ground_truth_lines=ground_truth_lines,
        raster_guess_reader=raster_guess_reader,
        cell_size_in_meters=cell_size_in_meters,
        aoi=aoi,
    )
    results = {}
    for metric in metrics:
        results[metric.__name__] = metric(y_ture, y_pred)
    return results


def get_binary_arrays(
    ground_truth_lines: gp.GeoDataFrame,
    raster_guess_reader: rasterio.DatasetReader,
    cell_size_in_meters: Optional[float] = None,
    aoi: Optional[gp.GeoDataFrame] = None,
):
    """
    This function calculates the two Tensors y_pred, y_true suitable for computation of loss or metrics functions.

    :param ground_truth_lines: A gp.GeoDataFrame object which contains LineString objects as shapes
                               representing the grid lines.
    :param raster_guess_reader: A rasterio.DatasetReader object which contains the raster of predicted grid lines.
                                Pixel values marked with 1 are considered a prediction of a grid line.
    :param cell_size_in_meters: The cell_size_in_meters parameter controls the size of one prediction in meters.
                                E.g. the original raster has a pixel size of 100m x 100m.
                                A cell_size of 1000m meters means that one prediction
                                is now the grouping of 100 original pixels.
                                This is done for both the ground truth raster and the prediction raster.
                                The down-sampling strategy considers a collection of pixel values as a positive
                                prediction (value = 1) if at least one pixel in that collection has the value 1.
    :param aoi: A gp.GeoDataFrame containing exactly one Polygon or Multipolygon marking the area of interest.
                The CRS is expected to be the same as the raster_guess_readers' CRS.

    :returns: y_pred, y_true: Two binary 1-D arrays of same length.
    """
    # perform clipping of raster and ground truth in case aoi parameter is provided
    if aoi is not None:
        ground_truth_lines = clip_line_poly(ground_truth_lines, aoi)
        raster, affine = get_clipped_data(raster_guess_reader, aoi)
        raster = raster.squeeze(axis=0)
    else:
        raster, affine = raster_guess_reader.read(1), raster_guess_reader.transform

    # perform down-sampling in case cell_size_in_meters parameter is provided.
    if cell_size_in_meters is not None:
        current_cell_size_x, current_cell_size_y = get_resolution_in_meters(
            raster_guess_reader
        )
        if current_cell_size_x != current_cell_size_y:
            raise ValueError(
                f"Only quadratic pixel values are supported for scaling."
                f" Found pixel size x {current_cell_size_x}"
                f" and pixel size y P{current_cell_size_y}."
            )
        scaling = current_cell_size_x / cell_size_in_meters
        if scaling > 1.0:
            raise ValueError(
                f"Up-sampling not supported. Select a cell size of at least {current_cell_size_x}."
            )
        raster, affine = _perform_scaling(
            raster, affine, scaling, crs=raster_guess_reader.crs.to_string()
        )
    raster_ground_truth = _rasterize_geo_dataframe(raster, ground_truth_lines, affine)
    return raster.flatten(), raster_ground_truth.flatten()


def _perform_scaling(
    raster_array: np.array, affine_mat: Affine, scaling_factor: float, crs: str
) -> Tuple[np.array, Affine]:
    shape = (
        1,
        round(raster_array.shape[0] * scaling_factor),
        round(raster_array.shape[1] * scaling_factor),
    )
    raster_out = np.empty(shape)

    raster_out_transform = affine_mat * affine_mat.scale(
        (raster_array.shape[0] / shape[1]), (raster_array.shape[1] / shape[2])
    )

    with fiona.Env():
        with rasterio.Env():
            reproject(
                raster_array,
                raster_out,
                src_transform=affine_mat,
                dst_transform=raster_out_transform,
                src_crs={"init": crs},
                dst_crs={"init": crs},
                resampling=Resampling.max,
            )
    return raster_out.squeeze(axis=0), raster_out_transform


def _rasterize_geo_dataframe(
    raster_array: np.array, data_frame: gp.GeoDataFrame, transform: Affine
) -> np.array:
    """All raster values where shapes are found or which are touched
    by a shape will have the values one, the rest zero."""
    assert (
        len(raster_array.shape) == 2
    ), f"Expected 2D array, got shape {raster_array.shape}."
    data_rows = [row.geometry for _, row in data_frame.iterrows()]
    new_raster = rasterize(
        data_rows,
        out_shape=raster_array.shape,
        fill=0,
        default_value=1,
        all_touched=True,
        transform=transform,
    )
    return new_raster
