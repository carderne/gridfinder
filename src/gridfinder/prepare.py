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
from scipy.ndimage import label
from pathlib import Path
from typing import Union, List


import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from gridfinder.util.raster import get_clipped_data
from gridfinder.electrificationfilter import ElectrificationFilter


def combine_rasters_into_single_file(
    rasters: List[Union[str, Path, rasterio.DatasetReader]],
    output_file: Union[str, Path],
) -> str:
    assert len(rasters) > 1, f"Need at least two rasters to combine. Got {len(rasters)}"
    crs = None
    for i, raster in enumerate(rasters):
        if isinstance(raster, (str, Path)):
            rasters[i] = rasterio.open(raster)
        crs = rasters[i].crs

    # TODO: Check save_2d_array_as_raster: When trying to save the raster_out array with that function
    #       there are some transparent pixels (checked in QGIS) which are not present when saving with this method
    out_raster, out_trans = merge(rasters)
    out_meta = rasters[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_raster.shape[1],
            "width": out_raster.shape[2],
            "transform": out_trans,
            "crs": crs,
        }
    )
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_raster)
    return output_file


def merge_rasters(file_paths: List[Union[str, Path]], percentile=70):
    """Merge a set of monthly rasters keeping the nth percentile value.

    Used to remove transient features from time-series data.

    :param file_paths: List of paths to raster files that are to be merged.
    :type file_paths: List[str], List[Path]
    :param percentile: Percentile value to use when merging using np.nanpercentile.
        Lower values will result in lower values/brightness. (Default value = 70)
    :type percentile: int, optional (default 70.)


    """

    affine = None
    crs = None
    rasters = []

    for file in file_paths:
        if file.endswith(".tif"):
            with rasterio.open(file) as ntl_rd:
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


def drop_zero_pop(
    targets_in: Union[str, Path],
    pop_in: Union[str, Path],
    aoi: Union[str, Path, gpd.GeoDataFrame],
) -> np.array:
    """
    This function first reprojects the clipped population dataset to match
    the resolution of the targets.
    Afterwards it finds connected components (pixels) in the target dataset.
    Then it checks if the connected component has population of bigger than one.
    If not, the target is set to zero for the whole component area.

    :param targets_in: Path to output from prepare_ntl()
    :param pop_in: Path to a population raster such as GHS or HRSL.
    :param aoi: An AOI to use to clip the population raster.

    :returns Processed Targets
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

    # neighbourhood filter
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(targets, structure)

    assert labeled.shape == targets.shape
    assert ghs_proj.shape == targets.shape

    for c in range(1, ncomponents + 1):
        connected_targets = np.nonzero(labeled == c)
        population_values = ghs_proj[connected_targets]
        population_over_one = np.any(population_values[population_values > 1])
        if not population_over_one:
            targets[connected_targets] = 0

    return targets


def prepare_roads(roads_in, aoi, ntl_in, power=None):
    """Prepare a roads feature layer for use in algorithm.

    :param roads_in: Path to a roads feature layer. This implementation is specific to
        OSM data and won't assign proper weights to other data inputs.
    :type roads_in: str, Path
    :param aoi: AOI to clip roads.
    :type aoi: str, Path or GeoDataFrame
    :param power: OSM power lines
    :type power: gp.GeoDataFrame
    :param ntl_in: Path to a raster file, only used for correct shape and
        affine of roads raster.
    :type ntl_in: str, Path

    """

    ntl_rd = rasterio.open(ntl_in)
    shape = ntl_rd.read(1).shape
    affine = ntl_rd.transform

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

    # ignore locations that already have a high voltage line
    if power is not None:
        roads = gpd.sjoin(roads, power, how="left", op="intersects")
        if "power_right" in roads.columns:
            roads.loc[roads["power_right"] == "line", "weight"] = 0
        elif "power" in roads.columns:
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
