"""
Prepare input layers for gridfinder.

Functions:

- clip_rasters
- merge_rasters
- filter_func
- create_filter
- prepare_ntl
- drop_zero_pop
- prepare_roads
"""

import os
from pathlib import Path
from typing import Union


import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from gridfinder.util.raster import get_clipped_data
from gridfinder.electrificationfilter import ElectrificationFilter


def merge_rasters(folder: Union[str, Path], percentile=70):
    """Merge a set of monthly rasters keeping the nth percentile value.

    Used to remove transient features from time-series data.

    :param folder: Folder containing rasters to be merged.
    :type folder: str, Path
    :param percentile: Percentile value to use when merging using np.nanpercentile.
        Lower values will result in lower values/brightness. (Default value = 70)
    :type percentile: int, optional (default 70.)


    """

    affine = None
    crs = None
    rasters = []

    for file in os.listdir(folder):
        if file.endswith(".tif"):
            with rasterio.open(os.path.join(folder, file)) as ntl_rd:
                rasters.append(ntl_rd.read(1))

                if not affine:
                    affine = ntl_rd.transform
                if not crs:
                    crs = ntl_rd.crs
    raster_arr = np.array(rasters)

    raster_merged = np.percentile(raster_arr, percentile, axis=0)

    return raster_merged, affine, crs


def prepare_ntl(
    ntl: np.ndarray,
    affine: gpd.GeoDataFrame,
    electrification_predictor: ElectrificationFilter,
    threshold=0.1,
    upsample_by=2,
):
    """Convert the supplied NTL raster and output an array of electrified cells
    as targets for the algorithm by applying an electrification predictor over it,
    upsample the result, and convert to binary values by applying threshold.

    :param ntl: The nightlight imagery.
    :type ntl: np.ndarray
    :param affine: The affine transformation.
    :type affine: gpd.GeoDataFrame
    :param electrification_predictor: The predictor is used to extract targets from the raster data
    :type electrification_predictor: numpy array
    :param threshold: The threshold to apply after filtering, values above
        are considered electrified. (Default value = 0.1)
    :type threshold: float, optional (default 0.1.)
    :param upsample_by: The factor by which to upsample the input raster, applied to both axes
        (so a value of 2 results in a raster 4 times bigger). This is to
        allow the roads detail to be captured in higher resolution. (Default value = 2)
    :type upsample_by: int, optional (default 2.)

    """
    ntl_filtered = electrification_predictor.predict(ntl)
    ntl_interp, newaff = _upsample(affine, ntl_filtered, upsample_by)
    ntl_thresh = (ntl_interp[0] >= threshold).astype(float)
    return ntl_thresh, newaff


def _upsample(affine: gpd.GeoDataFrame, ntl_filtered: np.ndarray, upsample_by: int):
    """
    Upsample the input raster and return upsampled raster and new affine transformation.
    :param affine: The input affine transformation.
    :type affine: gpd.GeoDataFrame
    :param ntl_filtered: Input raster.
    :type ntl_filtered: np.ndarray
    :param upsample_by: Factor to be applied to both axes,
        e.g. 2 will make the raster 4 times bigger.
    :type upsample_by: int

    """
    ntl_interp = np.empty(
        shape=(
            1,
            round(ntl_filtered.shape[0] * upsample_by),
            round(ntl_filtered.shape[1] * upsample_by),
        )
    )
    newaff = Affine(
        affine.a / upsample_by,
        affine.b,
        affine.c,
        affine.d,
        affine.e / upsample_by,
        affine.f,
    )
    with fiona.Env():
        with rasterio.Env():
            reproject(
                ntl_filtered,
                ntl_interp,
                src_transform=affine,
                dst_transform=newaff,
                src_crs={"init": "epsg:4326"},
                dst_crs={"init": "epsg:4326"},
                resampling=Resampling.bilinear,
            )
    return ntl_interp, newaff


def drop_zero_pop(targets_in, pop_in, aoi):
    """Drop electrified cells with no other evidence of human activity.

    :param targets_in: Path to output from prepare_ntl()
    :type targets_in: str, Path
    :param pop_in: Path to a population raster such as GHS or HRSL.
    :type pop_in: str, Path
    :param aoi: An AOI to use to clip the population raster.
    :type aoi: str, Path or GeoDataFrame


    """

    if isinstance(aoi, (str, Path)):
        aoi = gpd.read_file(aoi)

    # Clip population layer to AOI
    with rasterio.open(pop_in) as dataset:
        crs = dataset.crs
        clipped, affine = get_clipped_data(dataset, aoi)
    # clipped = clipped[0]  # no longer needed, fixed in clip_raster

    # We need to warp the population layer to exactly overlap with targets
    # First get array, affine and crs from targets (which is what we)
    targets_rd = rasterio.open(targets_in)
    targets = targets_rd.read(1)
    ghs_proj = np.empty_like(targets)
    dest_affine = targets_rd.transform
    dest_crs = targets_rd.crs

    # Then use reproject
    with rasterio.Env():
        reproject(
            source=clipped,
            destination=ghs_proj,
            src_transform=affine,
            dst_transform=dest_affine,
            src_crs=crs,
            dst_crs=dest_crs,
            resampling=Resampling.bilinear,
        )

    # Finally read to run algorithm to drop blobs (areas of target==1)
    # where there is no underlying population

    blobs = []
    skip = []
    max_i = targets.shape[0]
    max_j = targets.shape[1]

    def add_around(blob, cell):
        """

        :param blob:
        :param cell:

        """
        blob.append(cell)
        skip.append(cell)

        for x in range(-1, 2):
            for y in range(-1, 2):
                next_i = i + x
                next_j = j + y
                next_cell = (next_i, next_j)

                # ensure we're within bounds
                if next_i >= 0 and next_j >= 0 and next_i < max_i and next_j < max_j:
                    # ensure we're not looking at same spot or one that's done
                    if not next_cell == cell and next_cell not in skip:
                        # if it's an electrified cell
                        if targets[next_i][next_j] == 1:
                            blob = add_around(blob, next_cell)

        return blob

    for i in range(max_i):
        for j in range(max_j):
            if targets[i][j] == 1 and (i, j) not in skip:
                blob = add_around(blob=[], cell=(i, j))
                blobs.append(blob)

    for blob in blobs:
        found = False
        for cell in blob:
            if ghs_proj[cell] > 1:
                found = True
                break
        if not found:
            # set all values in blob to 0
            for cell in blob:
                targets[cell] = 0

    return targets


def prepare_roads(roads_in, aoi_in, ntl_in, include_power=True):
    """Prepare a roads feature layer for use in algorithm.

    :param roads_in: Path to a roads feature layer. This implementation is specific to
        OSM data and won't assign proper weights to other data inputs.
    :type roads_in: str, Path
    :param aoi_in: AOI to clip roads.
    :type aoi_in: str, Path or GeoDataFrame
    :param ntl_in: Path to a raster file, only used for correct shape and
        affine of roads raster.
    :type ntl_in: str, Path
    :param include_power:  (Default value = True)


    """

    ntl_rd = rasterio.open(ntl_in)
    shape = ntl_rd.read(1).shape
    affine = ntl_rd.transform

    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    roads_masked = gpd.read_file(roads_in, mask=aoi)
    roads = gpd.sjoin(roads_masked, aoi, how="inner", op="intersects")
    roads = roads[roads_masked.columns]

    roads["weight"] = 1
    roads.loc[roads["highway"] == "motorway", "weight"] = 1 / 10
    roads.loc[roads["highway"] == "trunk", "weight"] = 1 / 9
    roads.loc[roads["highway"] == "primary", "weight"] = 1 / 8
    roads.loc[roads["highway"] == "secondary", "weight"] = 1 / 7
    roads.loc[roads["highway"] == "tertiary", "weight"] = 1 / 6
    roads.loc[roads["highway"] == "unclassified", "weight"] = 1 / 5
    roads.loc[roads["highway"] == "residential", "weight"] = 1 / 4
    roads.loc[roads["highway"] == "service", "weight"] = 1 / 3

    # Power lines get weight 0
    if "power" in roads:
        roads.loc[roads["power"] == "line", "weight"] = 0

    roads = roads[roads.weight != 1]

    # sort by weight descending so that lower weight (bigger roads) are
    # processed last and overwrite higher weight roads
    roads = roads.sort_values(by="weight", ascending=False)

    roads_for_raster = [(row.geometry, row.weight) for _, row in roads.iterrows()]
    roads_raster = rasterize(
        roads_for_raster,
        out_shape=shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=affine,
    )

    return roads_raster, affine
