from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio as rs
from affine import Affine
from geopandas import GeoDataFrame
from pyproj import Proj
from rasterio.mask import mask

Pathy = str | Path

Loc = tuple[int, int]


def save_raster(
    path: Pathy,
    raster: np.ndarray,
    affine: Affine,
    crs: str | Proj | None = None,
    nodata: int = 0,
) -> None:
    """Save a raster to the specified file.

    Parameters

    ----------
    file : Output file path
    raster : 2D numpy array containing raster values
    affine : Affine transformation for the raster
    crs: CRS for the raster (default EPSG4326)
    """

    p_path = Path(path)
    if not p_path.parents[0].exists():
        p_path.parents[0].mkdir(parents=True, exist_ok=True)

    if not crs:
        crs = "+proj=latlong"

    with rs.open(
        p_path,
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=affine,
        nodata=nodata,
    ) as filtered_out:
        filtered_out.write(raster, 1)


def clip_raster(
    raster: Pathy,
    boundary: GeoDataFrame,
) -> tuple[
    np.ndarray,
    Affine,
    dict,
]:
    """Clip the raster to the given administrative boundary.

    Parameters
    ----------
    raster : Location of or already opened raster.
    boundary : The polygon by which to clip the raster.

    Returns
    -------
    clipped : Contents of clipped raster.
    affine : The affine
    crs : form {'init': 'epsg:4326'}
    """

    raster_ds = rs.open(raster)
    raster_crs = raster_ds.crs

    if not (boundary.crs == raster_crs or boundary.crs == raster_crs.data):
        boundary = boundary.to_crs(crs=raster_crs)  # type: ignore
    coords = [json.loads(boundary.to_json())["features"][0]["geometry"]]

    # mask/clip the raster using rasterio.mask
    clipped, affine = mask(dataset=raster_ds, shapes=coords, crop=True)

    if len(clipped.shape) >= 3:
        clipped = clipped[0]

    raster_ds.close()

    return clipped, affine, raster_crs
