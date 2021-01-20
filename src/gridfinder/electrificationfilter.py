import math
from math import sqrt
import abc
import numpy as np
from scipy import signal
from mullet.filters.models import TorchImageFilter, Conv2dImageFilter
import torch
from typing import Tuple


def get_weights_array(shape: Tuple[int, int]):
    """
    Construct a predictor from the pixel-wise function.
    The predictor is normalized so that and the output is then subtracted from the original image.
    :param shape: shape of filter
    :return: Numpy array with predictor values
    """

    def _filter_func(i: int, j: int):
        """
        d is that pixel’s perpendicular distance from a given square sample’s centroid.
        The goal of this predictor is to find pixels with a higher value than their neighborhood,
        biased towards closer areas. To achieve this bias, a non-linear function is needed,
        and a cubic function was found to achieve better results than a square function.

        :param i: row number with respect to predictor height
        :param j: column number with respect to predictor width
        """
        d_rows = abs(i - math.floor(shape[0] / 2))
        d_cols = abs(j - math.floor(shape[1] / 2))
        d = sqrt(d_rows ** 2 + d_cols ** 2)

        if d == 0:
            return 0.0
        else:
            return 1 / (1 + d / 2) ** 3

    vec_filter_func = np.vectorize(_filter_func)
    ntl_filter = np.fromfunction(vec_filter_func, shape, dtype=float)
    return ntl_filter / ntl_filter.sum()


class ElectrificationFilter(metaclass=abc.ABCMeta):
    """Abstract class for electrification filters. Electrification filters
    are filters that extract electrification targets from satellite imagery."""

    @abc.abstractmethod
    def predict(self, data: np.ndarray):
        pass


class NightlightFilter(ElectrificationFilter):
    """The electrification_predictor that was used in this Nature paper:
    https://www.nature.com/articles/s41597-019-0347-4"""

    def __init__(self, shape=(41, 41)):
        """
        :param shape: Shape of predictor, e.g. (2,2)
        """
        self.shape = shape
        self.predictor = get_weights_array(shape)

    def predict(self, data: np.ndarray):
        """
        Apply predictor on input data.
        :param data:
        :return:
        """
        ntl_convolved = signal.convolve2d(
            data, self.predictor, mode="same", boundary="symm"
        )
        ntl_filtered = data - ntl_convolved
        return ntl_filtered


class NightlightTorchFilter(Conv2dImageFilter):
    """
    Implementation of NightlightFilter in PyTorch.
    """

    def __init__(
        self, init_weights=get_weights_array, radius=21, bias=False, threshold=0.4
    ):
        super().__init__(radius=radius, bias=bias, padding_mode="reflect")
        self.threshold = threshold
        self._set_weight(init_weights((self.kernel_size, self.kernel_size)))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        conv_result = self.conv2(x)
        x = torch.sub(x, conv_result)
        x = torch.add(x, self.threshold)
        return x.squeeze()
