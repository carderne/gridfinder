import pytest
from numpy.testing import assert_array_equal
from gridfinder.post import threshold_distances
import numpy as np


@pytest.mark.parametrize(
    "raster, threshold, expected",
    [
        (
            [[[5.0, 5.0], [4.0, 4.0]], [[5.0, 5.0], [4.0, 4.0]]],
            2.0,
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ),
        ([2.0], 3.0, [1.0]),
    ],
)
def test_threshold_distances(raster, threshold, expected):
    assert_array_equal(
        threshold_distances(np.array(raster), threshold), np.array(expected)
    )
