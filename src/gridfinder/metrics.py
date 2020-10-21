from typing import Optional

import fiona
from dataclasses import dataclass

from affine import Affine
import numpy as np
import geopandas as gp
import rasterio
from rasterio.enums import Resampling
import rasterio.warp
from rasterio.warp import reproject
from rasterio.features import rasterize

from gridfinder._util import clip_line_poly
from gridfinder.util.raster import get_clipped_data, get_resolution_in_meters


@dataclass()
class ConfusionMatrix:
    tp: float = 0.0
    fp: float = 0.0
    tn: float = 0.0
    fn: float = 0.0


def confusion_matrix(
    ground_truth_lines: gp.GeoDataFrame,
    raster_guess_reader: rasterio.DatasetReader,
    cell_size_in_meters: Optional[float] = None,
    aoi: Optional[gp.GeoDataFrame] = None,
):
    """
    Calculates the
     - true positives
     - true negatives
     - false positives
     - false negatives
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

    :returns: ConfusionMatrix

    """

    def get_scaling_factor(desired_cell_size: float, current_cell_size: float) -> float:
        return current_cell_size / desired_cell_size

    def perform_scaling(
        raster_array: np.array, affine_mat: Affine, scaling_factor: float, crs: str
    ) -> np.array:
        shape = (
            1,
            round(raster_array.shape[0] * scaling_factor),
            round(raster_array.shape[1] * scaling_factor),
        )
        raster_out = np.empty(shape)

        raster_out_transform = affine_mat * affine_mat.scale(
            (raster_array.shape[0] / shape[0]), (raster_array.shape[1] / shape[1])
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

    def rasterize_geo_dataframe(
        raster_array: np.array, data_frame: gp.GeoDataFrame, transform: Affine
    ) -> np.array:
        """ All raster values where shapes are found will have the values one, the rest zero."""
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

    def measure_cell_wise_classification(
        ground_truth: np.array, prediction: np.array
    ) -> ConfusionMatrix:
        assert (
            len(ground_truth.shape) == 2
        ), f"Expected 2d array but got {ground_truth.shape}"
        assert ground_truth.shape == prediction.shape, (
            f"Ground truth and prediction have unequal shape."
            f" {ground_truth.shape} vs {prediction.shape}"
        )
        num_rows, num_columns = ground_truth.shape
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i in range(num_rows):
            for j in range(num_columns):
                predicted_value = int(prediction[i, j])
                actual_value = int(ground_truth[i, j])

                if actual_value == predicted_value == 1:
                    true_positives += 1
                if predicted_value == 1 and actual_value != predicted_value:
                    false_positives += 1
                if actual_value == predicted_value == 0:
                    true_negatives += 1
                if predicted_value == 0 and predicted_value != actual_value:
                    false_negatives += 1

        return ConfusionMatrix(
            tp=true_positives, fp=false_positives, tn=true_negatives, fn=false_negatives
        )

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
        scaling = get_scaling_factor(cell_size_in_meters, current_cell_size_x)
        if scaling > 1.0:
            raise ValueError(
                f"Up-sampling not supported. Select a cell size of at least {current_cell_size_x}."
            )
        raster, affine = perform_scaling(
            raster, affine, scaling, crs=raster_guess_reader.crs.to_string()
        )

    raster_ground_truth = rasterize_geo_dataframe(raster, ground_truth_lines, affine)
    return measure_cell_wise_classification(raster_ground_truth, raster)
