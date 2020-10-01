import logging
import os
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio import DatasetReader
from rasterio.mask import mask
from shapely.geometry import Polygon

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
    :return: Tuple containing the clipped and cropped data as numpy array and the associated affine transformation
    """
    if len(boundary_geodf) != 1:
        raise ValueError(
            f"Expected geo data frame with exactly one row, instead got {len(boundary_geodf)}"
        )

    if boundary_geodf.crs != dataset.crs:
        log.info(
            f"crs mismatch, projecting boundary from {boundary_geodf.crs} to {dataset.crs}"
        )
        boundary_geodf = boundary_geodf.to_crs(dataset.crs)

    boundary_area = boundary_geodf.geometry[0]
    if not isinstance(boundary_area, Polygon):
        raise ValueError(
            f"Expected geometry to be a Polygon, instead got {boundary_area.__class__}"
        )

    return mask(dataset=dataset, shapes=boundary_area, crop=True, nodata=nodata)
