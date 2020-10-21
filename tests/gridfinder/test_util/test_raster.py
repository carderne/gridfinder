import logging
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio import DatasetReader
from shapely.geometry import Point, Polygon, MultiPolygon

from gridfinder.util.raster import save_2d_array_as_raster, get_clipped_data


@pytest.fixture()
def output_file(tmpdir_factory):
    tmpfolder = tmpdir_factory.mktemp("tmp_raster")
    return os.path.join(tmpfolder, "sample_raster.tif")


@pytest.fixture()
def sample_polygon_geodf(test_resources) -> gpd.GeoDataFrame:
    return gpd.read_file(os.path.join(test_resources, "sample_Polygon.geojson"))


@pytest.fixture()
def sample_mul_polygon_geodf(test_resources) -> gpd.GeoDataFrame:
    polygons_geodf = gpd.read_file(
        os.path.join(test_resources, "sample_MultiPolygon.geojson")
    )
    multi_polygon = MultiPolygon(polygons_geodf.geometry.values)
    return gpd.GeoDataFrame({"geometry": [multi_polygon]}, crs=polygons_geodf.crs)


def test_save_2d_array_as_raster_inputValidation(
    three_dim_array, sample_transform, output_file, default_crs
):
    with pytest.raises(ValueError):
        save_2d_array_as_raster(
            output_file, three_dim_array, sample_transform, default_crs
        )


def test_save_2d_array_as_raster_savedArrayIsCorrect(
    two_dim_array, sample_transform, output_file, default_crs
):
    save_2d_array_as_raster(output_file, two_dim_array, sample_transform, default_crs)
    with rasterio.open(output_file) as dataset:
        dataset: DatasetReader
        data, transform, crs = dataset.read(), dataset.transform, dataset.crs
        assert np.allclose(two_dim_array, data)
        assert sample_transform == transform
        assert dataset.crs.to_string() == default_crs


def test_get_clipped_data_inputValidation(sample_ntl_raster, caplog):
    point_geodf = gpd.GeoDataFrame({"geometry": [Point(1.0, -1.0)]})
    coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    too_long_geodf = gpd.GeoDataFrame({"geometry": [Polygon(coords), Polygon(coords)]})
    with pytest.raises(ValueError) as e_info:
        get_clipped_data(sample_ntl_raster, point_geodf)
        assert "Expected geo data frame with exactly one row" in e_info.value
    with pytest.raises(ValueError) as e_info:
        get_clipped_data(sample_ntl_raster, too_long_geodf)
        assert "Point" in e_info.value


def test_get_clipped_data_crsGetsAdjusted(
    sample_ntl_raster, sample_polygon_geodf, caplog
):
    sample_polygon_geodf.to_crs(crs="EPSG:3785", inplace=True)
    with caplog.at_level(logging.INFO):
        clipped_data, _ = get_clipped_data(sample_ntl_raster, sample_polygon_geodf)
        full_data = sample_ntl_raster.read()
        assert "crs mismatch" in caplog.text
        assert clipped_data.shape < full_data.shape


@pytest.mark.parametrize(
    ["boundary"],
    [
        (pytest.lazy_fixture("sample_polygon_geodf"),),
        (pytest.lazy_fixture("sample_mul_polygon_geodf"),),
    ],
)
def test_get_clipped_data_clippingWorks(boundary, sample_ntl_raster, caplog):
    clipped_data, _ = get_clipped_data(sample_ntl_raster, boundary)
    full_data = sample_ntl_raster.read()
    assert clipped_data.shape < full_data.shape
