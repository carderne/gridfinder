from math import sqrt
from pathlib import Path
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

import numpy as np
from scipy import signal

import rasterio
from rasterio.mask import mask
from rasterio.features import shapes, rasterize
from rasterio import Affine
from rasterio.warp import reproject, Resampling

import geopandas as gpd

from gridfinder._util import save_raster, clip_line_poly


def threshold(dists_in, cutoff=0.5):
    """

    """
    dists_rd = rasterio.open(dists_in)
    dists_r = dists_rd.read(1)

    guess = np.empty_like(dists_r)
    guess[:] = dists_r[:]

    guess[dists_r >= cutoff] = 0
    guess[dists_r < cutoff] = 1
    
    return dists_r, guess, dists_rd.transform


def guess2geom(guess_in):
    """

    """

    guess_rd = rasterio.open(guess_in)
    guess_r = guess_rd.read(1)
    transform = guess_rd.transform

    guess_geojson = {
        'type': 'FeatureCollection',
        'features': []
    }

    guess_features = shapes(guess_r, transform=transform)
    for f, v in guess_features:    
        guess_geojson['features'].append({
            'type': 'Feature',
            'properties': {
                'val': v
            },
            'geometry': f
        })

    guess_gdf = gpd.GeoDataFrame.from_features(guess_geojson, crs={'init': 'epsg:4326'})

    guess_gdf = guess_gdf.loc[guess_gdf['val'] == 1]

    return guess_r, guess_geojson, guess_gdf

def accuracy(grid_in, guesses_in, aoi_in, buffer_amount=0.01):
    """

    """

    if isinstance(aoi_in, gpd.GeoDataFrame):
        aoi = aoi_in
    else:
        aoi = gpd.read_file(aoi_in)

    grid = gpd.read_file(grid_in)
    grid_clipped = clip_line_poly(grid, aoi)
    grid_buff = grid_clipped.buffer(buffer_amount)

    guesses_reader = rasterio.open(guesses_in)
    guesses = guesses_reader.read(1)

    grid_for_raster = [(row.geometry) for _, row in grid_clipped.iterrows()]
    grid_raster = rasterize(grid_for_raster, out_shape=guesses_reader.shape, fill=1,
                         default_value=0, all_touched=True, transform=guesses_reader.transform)
    grid_buff_raster = rasterize(grid_buff, out_shape=guesses_reader.shape, fill=1,
                         default_value=0, all_touched=True, transform=guesses_reader.transform)

    grid_raster = flip_arr_values(grid_raster)
    grid_buff_raster = flip_arr_values(grid_buff_raster)

    tp = true_positives(guesses, grid_buff_raster)
    fn = false_negatives(guesses, grid_raster)

    return tp, fn, 


def true_positives(guesses, truths):
    """

    """
    yes_guesses = 0
    yes_guesses_correct = 0
    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x,y]
            truth = truths[x,y]
            if guess == 1:
                yes_guesses += 1
                if guess == truth:
                    yes_guesses_correct += 1

    return yes_guesses_correct / yes_guesses


def false_negatives(guesses, truths):
    """

    """
    actual_grid = 0
    actual_grid_missed = 0

    rows = guesses.shape[0]
    cols = guesses.shape[1]

    for x in range(0, rows):
        for y in range(0, cols):
            guess = guesses[x,y]
            truth = truths[x,y]
            
            if truth == 1:
                actual_grid += 1
                if guess != truth:
                    found = False
                    for i in range(-5,6):
                        for j in range(-5,6):
                            if i == 0 and j == 0:
                                continue
                            
                            shift_x = x+i
                            shift_y = y+j
                            if shift_x < 0 or shift_y < 0 or shift_x >= rows or shift_y >= cols:
                                continue
            
                            other_guess = guesses[shift_x, shift_y]
                            if other_guess == 1:
                                found = True
                    if not found:
                        actual_grid_missed += 1

    return actual_grid_missed / actual_grid


def flip_arr_values(arr):
    """

    """
    arr[arr == 1] = 2
    arr[arr == 0] = 1
    arr[arr == 2] = 0
    return arr