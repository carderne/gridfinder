import os
from pathlib import Path

import numpy as np

import rasterio
import geopandas as gpd

def save_raster(file, raster, affine):
   """

   """

   if not os.path.exists(os.path.dirname(file)):
    try:
        os.makedirs(os.path.dirname(file))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

   filtered_out = rasterio.open(file, 'w', driver='GTiff',
                                  height=raster.shape[0], width=raster.shape[1],
                                     count=1, dtype=raster.dtype,
                                  crs='+proj=latlong', transform=affine)
   filtered_out.write(raster, 1)
   filtered_out.close()

def clip_line_poly(shp, clip_obj):
    '''
    docs
    '''

    # Create a single polygon object for clipping
    poly = clip_obj.geometry.unary_union
    spatial_index = shp.sindex

    # Create a box for the initial intersection
    bbox = poly.bounds
    # Get a list of id's for each road line that overlaps the bounding box and subset the data to just those lines
    sidx = list(spatial_index.intersection(bbox))
    shp_sub = shp.iloc[sidx]

    # Clip the data - with these data
    clipped = shp_sub.copy()
    clipped['geometry'] = shp_sub.intersection(poly)

    # Return the clipped layer with no null geometry values
    return(clipped[clipped.geometry.notnull()])