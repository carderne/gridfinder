"""
Utility module used internally.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from geopandas import GeoDataFrame
from pyproj import Proj
from rasterio.io import DatasetReader
from rasterio.mask import mask


def save_raster(
    path: Union[str, Path],
    raster: np.ndarray,
    affine: Affine,
    crs: Optional[Union[str, Proj]] = None,
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

    filtered_out = rasterio.open(
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
    )
    filtered_out.write(raster, 1)
    filtered_out.close()


def clip_raster(
    raster: Union[str, Path, DatasetReader],
    boundary: Union[str, Path, GeoDataFrame],
    boundary_layer: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    Affine,
    dict,
]:
    """Clip the raster to the given administrative boundary.

    Parameters
    ----------
    raster : Location of or already opened raster.
    boundary : The polygon by which to clip the raster.
    boundary_layer : For multi-layer files (like GeoPackage), specify layer to be used.

    Returns
    -------
    clipped : Contents of clipped raster.
    affine : The affine
    crs : form {'init': 'epsg:4326'}
    """

    raster_ds = raster if isinstance(raster, DatasetReader) else rasterio.open(raster)

    boundary_gdf: GeoDataFrame = (
        boundary
        if isinstance(boundary, GeoDataFrame)
        else gpd.read_file(boundary, layer=boundary_layer)
    )

    if not (
        boundary_gdf.crs == raster_ds.crs or boundary_gdf.crs == raster_ds.crs.data
    ):
        boundary_gdf = boundary_gdf.to_crs(crs=raster_ds.crs)  # type: ignore
    coords = [json.loads(boundary_gdf.to_json())["features"][0]["geometry"]]

    # mask/clip the raster using rasterio.mask
    clipped, affine = mask(dataset=raster_ds, shapes=coords, crop=True)

    if len(clipped.shape) >= 3:
        clipped = clipped[0]

    return clipped, affine, raster_ds.crs
