import os

import numpy as np
import pytest
import rasterio
from rasterio import DatasetReader

from gridfinder.util.raster import save_2d_array_as_raster


@pytest.fixture()
def output_file(tmpdir_factory):
    tmpfolder = tmpdir_factory.mktemp("tmp_raster")
    return os.path.join(tmpfolder, "sample_raster.tif")


def test_save_2d_array_as_raster_inputValidation(
    three_dim_array, sample_transform, output_file
):
    with pytest.raises(ValueError):
        save_2d_array_as_raster(output_file, three_dim_array, sample_transform)


def test_save_2d_array_as_raster_savedArrayIsCorrect(
    two_dim_array, sample_transform, output_file
):
    save_2d_array_as_raster(output_file, two_dim_array, sample_transform)
    with rasterio.open(output_file) as dataset:
        dataset: DatasetReader
        data, transform = dataset.read(), dataset.transform
        assert np.allclose(two_dim_array, data)
        assert sample_transform == transform


# TODO: test get_clipped_data input validation once we have finalized the interface and set up sample data
