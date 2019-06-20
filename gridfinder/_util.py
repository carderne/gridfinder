"""
Utility module used internally.

Functions:

- save_raster
- clip_line_poly
- clip_raster
"""

from pathlib import Path
import json

import geopandas as gpd
import rasterio
from rasterio.mask import mask


def save_raster(path, raster, affine, crs=None, nodata=0):
    """Save a raster to the specified file.

    Parameters
    ----------
    file : str
        Output file path
    raster : numpy.array
        2D numpy array containing raster values
    affine: affine.Affine
        Affine transformation for the raster
    crs: str, proj.Proj, optional (default EPSG4326)
        CRS for the raster
    """

    path = Path(path)
    if not path.parents[0].exists():
        path.parents[0].mkdir(parents=True, exist_ok=True)

    if not crs:
        crs = "+proj=latlong"

    filtered_out = rasterio.open(
        path,
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


def clip_line_poly(line, clip_poly):
    """Clip a line features by the provided polygon feature.

    Parameters
    ----------
    line : GeoDataFrame
        The line features to be clipped.
    clip_poly : GeoDataFrame
        The polygon used to clip the line.

    Returns
    -------
    clipped : GeoDataFrame
        The clipped line feature.
    """

    # Create a single polygon object for clipping
    poly = clip_poly.geometry.unary_union
    spatial_index = line.sindex

    # Create a box for the initial intersection
    bbox = poly.bounds
    # Get a list of id's for each road line that overlaps the bounding box
    # and subset the data to just those lines
    sidx = list(spatial_index.intersection(bbox))
    shp_sub = line.iloc[sidx]

    # Clip the data - with these data
    clipped = shp_sub.copy()
    clipped["geometry"] = shp_sub.intersection(poly)
    # remove null geometry values
    clipped = clipped[clipped.geometry.notnull()]

    return clipped


# clip_raster is copied from openelec.clustering
def clip_raster(raster, boundary, boundary_layer=None):
    """Clip the raster to the given administrative boundary.

    Parameters
    ----------
    raster : string, pathlib.Path or rasterio.io.DataSetReader
        Location of or already opened raster.
    boundary : string, pathlib.Path or geopandas.GeoDataFrame
        The polygon by which to clip the raster.
    boundary_layer : string, optional
        For multi-layer files (like GeoPackage), specify the layer to be used.


    Returns
    -------
    tuple
        Three elements:
            clipped : numpy.ndarray
                Contents of clipped raster.
            affine : affine.Affine()
                Information for mapping pixel coordinates
                to a coordinate system.
            crs : dict
                Dict of the form {'init': 'epsg:4326'} defining the coordinate
                reference system of the raster.

    """

    if isinstance(raster, Path):
        raster = str(raster)
    if isinstance(raster, str):
        raster = rasterio.open(raster)

    if isinstance(boundary, Path):
        boundary = str(boundary)
    if isinstance(boundary, str):
        if ".gpkg" in boundary:
            driver = "GPKG"
        else:
            driver = None  # default to shapefile
            boundary_layer = ""  # because shapefiles have no layers

        boundary = gpd.read_file(boundary, layer=boundary_layer, driver=driver)

    if not (boundary.crs == raster.crs or boundary.crs == raster.crs.data):
        boundary = boundary.to_crs(crs=raster.crs)
    coords = [json.loads(boundary.to_json())["features"][0]["geometry"]]

    # mask/clip the raster using rasterio.mask
    clipped, affine = mask(dataset=raster, shapes=coords, crop=True)

    if len(clipped.shape) >= 3:
        clipped = clipped[0]

    return clipped, affine, raster.crs
