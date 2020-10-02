import logging
import os
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio import DatasetReader
from rasterio.mask import mask
from shapely.geometry import Polygon, MultiPolygon

log = logging.getLogger(__name__)


def save_2d_array_as_raster(
    path: str, arr: np.ndarray, transform: Affine, crs=None, nodata=0
):
    if not len(arr.shape) == 2:
        raise ValueError(f"Expected 2-dim array, instead got shape: {arr.shape}")
    log.debug(f"Saving raster to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as file:
        file.write(arr, 1)


def get_clipped_data(
    dataset: DatasetReader, boundary_geodf: gpd.GeoDataFrame, nodata=0
) -> Tuple[np.ndarray, Affine]:
    """

    :param dataset: opened raster
    :param boundary_geodf: a GeoDataFrame with exactly one row containing the boundary.
        Currently supported boundary types are Polygon and MultiPolygon
    :param nodata:
    :return: Tuple containing the clipped and cropped data as numpy array and the associated affine transformation
    """
    if len(boundary_geodf) != 1:
        raise ValueError(
            f"Expected geo data frame with exactly one row, instead got {len(boundary_geodf)}"
        )

    dataset_crs = dataset.crs.to_string()
    if boundary_geodf.crs is not None:
        boundary_crs = boundary_geodf.crs.to_string()
    else:
        boundary_crs = None
    if boundary_crs != dataset_crs:
        log.info(
            f"crs mismatch, projecting boundary from {boundary_geodf.crs} to {dataset.crs}"
        )
        boundary_geodf = boundary_geodf.to_crs(dataset_crs)

    boundary_shape = boundary_geodf.geometry[0]
    if not isinstance(boundary_shape, Polygon) and not isinstance(
        boundary_shape, MultiPolygon
    ):
        raise ValueError(f"Unsupported geometry: {boundary_shape.__class__}")
    # often the boundary will be a multipolygon, e.g. for a country. In such cases, we take the hull as boundary shape
    boundary_shape = boundary_shape.convex_hull

    return mask(dataset=dataset, shapes=[boundary_shape], crop=True, nodata=nodata)
