# gridfinder
# Licensed under GPL-3.0
# (C) Christopher Arderne

import os
from pathlib import Path
import json

import geopandas as gpd
import rasterio
from rasterio.mask import mask


def save_raster(file, raster, affine, crs=None):
    """

    """
    if not os.path.exists(os.path.dirname(file)):
        try:
            os.makedirs(os.path.dirname(file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if not crs:
        crs = '+proj=latlong'

    filtered_out = rasterio.open(file, 'w', driver='GTiff',
                                 height=raster.shape[0], width=raster.shape[1],
                                 count=1, dtype=raster.dtype,
                                 crs=crs, transform=affine)
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


# clip_raster is copied from openelec.clustering
def clip_raster(raster, boundary, boundary_layer=None):
    """
    Clip the raster to the given administrative boundary.

    Parameters
    ----------
    raster: string, pathlib.Path or rasterio.io.DataSetReader
        Location of or already opened raster.
    boundary: string, pathlib.Path or geopandas.GeoDataFrame
        The poylgon by which to clip the raster.
    boundary_layer: string, optional
        For multi-layer files (like GeoPackage), specify the layer to be used.


    Returns
    -------
    tuple
        Three elements:
            clipped: numpy.ndarray
                Contents of clipped raster.
            affine: affine.Affine()
                Information for mapping pixel coordinates
                to a coordinate system.
            crs: dict
                Dict of the form {'init': 'epsg:4326'} defining the coordinate
                reference system of the raster.

    """

    if isinstance(raster, Path):
        raster = str(raster)
    if isinstance(raster, str):
        raster = rasterio.open(raster)

    crs = raster.crs
    
    if isinstance(boundary, Path):
        boundary = str(boundary)
    if isinstance(boundary, str):
        if '.gpkg' in boundary:
            driver = 'GPKG'
        else:
            driver = None  # default to shapefile
            boundary_layer = ''  # because shapefiles have no layers
    
        boundary = gpd.read_file(boundary, layer=boundary_layer, driver=driver)

    boundary = boundary.to_crs(crs=raster.crs)
    coords = [json.loads(boundary.to_json())['features'][0]['geometry']]

    # mask/clip the raster using rasterio.mask
    clipped, affine = mask(dataset=raster, shapes=coords, crop=True)

    return clipped, affine, crs