"""
Post-processing for gridfinder package.

Functions:

- threshold
- thin
- raster_to_lines
- accuracy
- true_positives
- false_negatives
- flip_arr_values
"""

from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
from skimage.morphology import skeletonize
import shapely.wkt
from shapely.geometry import Point, LineString
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy

from gridfinder._util import clip_line_poly


def _read_raster(filepath, raster_bands: Optional[Union[int, List[int]]]):
    """
    Read a raster file and return its content and affine transformation.

    Parameters
    ----------
    filepath: path-like object

    Returns
    -------
    file_read: numpy array
        Representation of raster file, either 2D if one band or 3D if multiple
    transform: numpy array
        The affine transformation of the raster file.
    crs: rasterio.crs.CRS
        Coordinate reference system
    """
    with rasterio.open(filepath) as file:
        file_read = file.read(raster_bands)
        transform = file.transform
        crs = file.crs
        return file_read, transform, crs


def read_and_threshold_distances(dists_in: Union[str, np.ndarray], threshold=0.0):
    """Convert distance array into binary array of connected locations.

    Parameters
    ----------
    dists_in : path-like or numpy array
        2D array output from gridfinder algorithm.
    threshold : float, optional (default 0.5.)
        Cutoff value below which consider the cells to be grid.

    Returns
    -------
    guess : numpy array
        Binary representation of input array.
    """
    if isinstance(dists_in, (str, Path)):
        dists, _, _ = _read_raster(dists_in, 1)
        return _threshold_array(dists, threshold)

    elif isinstance(dists_in, np.ndarray):
        return _threshold_array(dists_in, threshold)

    else:
        raise ValueError("Please provide either a file path or a numpy.ndarray.")


def _threshold_array(arr, threshold):
    """Return an array where values are 1. if smaller than threshold,
    and 0. otherwise"""
    return (arr <= threshold).astype(float)


def thin(guess_in):
    """
    Use scikit-image skeletonize to 'thin' the guess raster.

    Parameters
    ----------
    guess_in : path-like or 2D array
        Output from threshold().

    Returns
    -------
    guess_skel : numpy array
        Thinned version.
    """

    if isinstance(guess_in, (str, Path)):
        guess, _, _ = _read_raster(guess_in, 1)
        guess_skel = skeletonize(guess)

    elif isinstance(guess_in, np.ndarray):
        guess_skel = skeletonize(guess_in)

    else:
        raise ValueError("Please provide either a file path or a numpy.ndarray.")

    return guess_skel.astype("int32")


def raster_to_lines(guess_skel_in):
    """
    Convert thinned raster to linestring geometry.

    Parameters
    ----------
    guess_skel_in : path-like
        Output from thin().

    Returns
    -------
    guess_gdf : GeoDataFrame
        Converted to geometry.
    """

    arr, affine, crs = _read_raster(guess_skel_in, 1)

    max_row = arr.shape[0]
    max_col = arr.shape[1]
    lines = []

    for row in range(0, max_row):
        for col in range(0, max_col):
            loc = (row, col)
            if arr[loc] == 1:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        next_row = row + i
                        next_col = col + j
                        next_loc = (next_row, next_col)

                        # ensure we're within bounds
                        # ensure we're not looking at the same spot
                        if (
                            next_row < 0
                            or next_col < 0
                            or next_row >= max_row
                            or next_col >= max_col
                            or next_loc == loc
                        ):
                            continue

                        if arr[next_loc] == 1:
                            line = (loc, next_loc)
                            rev = (line[1], line[0])
                            if line not in lines and rev not in lines:
                                lines.append(line)

    real_lines = []
    for line in lines:
        real = (xy(affine, line[0][0], line[0][1]), xy(affine, line[1][0], line[1][1]))
        real_lines.append(real)

    shapes = []
    for line in real_lines:
        shapes.append(LineString([Point(line[0]), Point(line[1])]).wkt)

    guess_gdf = pd.DataFrame(shapes)
    geometry = guess_gdf[0].map(shapely.wkt.loads)
    guess_gdf = guess_gdf.drop(0, axis=1)
    guess_gdf = gpd.GeoDataFrame(guess_gdf, crs=crs, geometry=geometry)

    guess_gdf["same"] = 0
    guess_gdf = guess_gdf.dissolve(by="same")
    guess_gdf = guess_gdf.to_crs(epsg=4326)

    return guess_gdf


def accuracy(grid_in, guess_in, aoi_in, buffer_amount=0.01):
    """Measure accuracy against a specified grid 'truth' file.

    Parameters
    ----------
    grid_in : str, Path
        Path to vector truth file.
    guess_in : str, Path
        Path to guess output from guess2geom.
    aoi_in : str, Path
        Path to AOI feature.
    buffer_amount : float, optional (default 0.01.)
        Leeway in decimal degrees in calculating equivalence.
        0.01 DD equals approximately 1 mile at the equator.
    """

    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    grid = gpd.read_file(grid_in)
    grid_clipped = clip_line_poly(grid, aoi)
    grid_buff = grid_clipped.buffer(buffer_amount)

    guesses_reader = rasterio.open(guess_in)
    guesses = guesses_reader.read(1)

    grid_for_raster = [(row.geometry) for _, row in grid_clipped.iterrows()]
    grid_raster = rasterize(
        grid_for_raster,
        out_shape=guesses_reader.shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=guesses_reader.transform,
    )
    grid_buff_raster = rasterize(
        grid_buff,
        out_shape=guesses_reader.shape,
        fill=1,
        default_value=0,
        all_touched=True,
        transform=guesses_reader.transform,
    )

    grid_raster = flip_arr_values(grid_raster)
    grid_buff_raster = flip_arr_values(grid_buff_raster)

    tp = true_positives(guesses, grid_buff_raster)
    fn = false_negatives(guesses, grid_raster)

    return tp, fn


def true_positives(guesses, truths):
    """Calculate true positives, used by accuracy().

    Parameters
    ----------
    guesses : numpy array
        Output from model.
    truths : numpy array
        Truth feature converted to array.

    Returns
    -------
    tp : float
        Ratio of true positives.
    """

    yes_guesses = 0
    yes_guesses_correct = 0
    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x, y]
            truth = truths[x, y]
            if guess == 1:
                yes_guesses += 1
                if guess == truth:
                    yes_guesses_correct += 1

    tp = yes_guesses_correct / yes_guesses

    return tp


def false_negatives(guesses, truths):
    """Calculate false negatives, used by accuracy().

    Parameters
    ----------
    guesses : numpy array
        Output from model.
    truths : numpy array
        Truth feature converted to array.

    Returns
    -------
    fn : float
        Ratio of false negatives.
    """

    actual_grid = 0
    actual_grid_missed = 0

    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x, y]
            truth = truths[x, y]

            if truth == 1:
                actual_grid += 1
                if guess != truth:
                    found = False
                    for i in range(-5, 6):
                        for j in range(-5, 6):
                            if i == 0 and j == 0:
                                continue

                            shift_x = x + i
                            shift_y = y + j
                            if shift_x < 0 or shift_y < 0:
                                continue
                            if shift_x >= rows or shift_y >= cols:
                                continue

                            other_guess = guesses[shift_x, shift_y]
                            if other_guess == 1:
                                found = True
                    if not found:
                        actual_grid_missed += 1

    fn = actual_grid_missed / actual_grid

    return fn


def flip_arr_values(arr):
    """Simple helper function used by accuracy()"""

    arr[arr == 1] = 2
    arr[arr == 0] = 1
    arr[arr == 2] = 0
    return arr
