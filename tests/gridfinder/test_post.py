import pytest
from numpy.testing import assert_array_equal
from gridfinder.post import read_and_threshold_distances
import numpy as np


@pytest.mark.parametrize(
    "raster, cutoff, expected",
    [
        (
            [[[5.0, 5.0], [4.0, 4.0]], [[5.0, 5.0], [4.0, 4.0]]],
            2.0,
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ),
        ([2.0], 3.0, [1.0]),
    ],
)
def test_read_and_threshold_distances(raster, cutoff, expected):
    assert_array_equal(
        read_and_threshold_distances(np.array(raster), cutoff), np.array(expected)
    )
