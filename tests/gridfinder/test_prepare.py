import pytest
import rasterio
from affine import Affine
from numpy.testing import assert_array_almost_equal
from gridfinder.prepare import merge_rasters
import numpy as np

# there is a problem when we use the exact identity matrix for an affine
# see: https://github.com/mapbox/rasterio/issues/674

TRANSFORM = Affine(1 + 1e-10, 0.0, 0.0, 0.0, 1 + 1e-10, 0.0)


def _store_tif_file(fn, raster):
    with rasterio.open(
        str(fn),
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs="EPSG:4326",
        transform=TRANSFORM,
    ) as dst:
        dst.write(raster, 1)
        dst.close()
    return str(fn)


class TestPrepare:
    @pytest.mark.parametrize(
        "tif_file, expected",
        [
            (
                [[[5.0, 5.0], [4.0, 4.0]], [[5.0, 5.0], [4.0, 4.0]]],
                [[5.0, 5.0], [4.0, 4.0]],
            ),
            (
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                [[1.0, 1.0], [1.0, 1.0]],
            ),
        ],
    )
    def test_merge_rasters(self, tmpdir_factory, tif_file, expected):
        tmpfolder = tmpdir_factory.mktemp("tmp")
        for key, val in enumerate(tif_file):
            fn = tmpfolder.join("tmp_" + str(key) + ".tif")
            _store_tif_file(fn, np.array(val))
        merged, affine = merge_rasters(tmpfolder)
        assert_array_almost_equal(np.array(expected), merged)
        assert TRANSFORM == affine
