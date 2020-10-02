"""
Utility module used internally.

Functions:

- save_raster
- clip_line_poly
- clip_raster
"""
import logging

log = logging.getLogger(__name__)


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
    clipped = clipped[~clipped.is_empty]

    return clipped
