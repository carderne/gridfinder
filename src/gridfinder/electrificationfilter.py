import math
from math import sqrt
import abc
import numpy as np
from scipy import signal
from mullet.filters.models import TorchImageFilter, Conv2dImageFilter
import torch
import torch.nn as nn
from typing import Tuple


def default_nightlight_target_filter(radius: int) -> np.ndarray:
    """
    Construct the weights for the default nightlight filter
    from the paper: https://www.nature.com/articles/s41597-019-0347-4
    The weights of the predictor are normalized and the output
    of the convolutional operation with these weights is then subtracted from the original image.
    This subtraction is accomplished by subtracting the weights from a 0-array with 1.0 in the center pixel.
    :param radius: radius of quadratic kernel
    :return: Numpy array with predictor values
    """

    def _filter_func(i: int, j: int) -> float:
        """
        d is that pixel’s perpendicular distance from a given square sample’s centroid.
        The goal of this predictor is to find pixels with a higher value than their neighborhood,
        biased towards closer areas. To achieve this bias, a non-linear function is needed,
        and a cubic function was found to achieve better results than a square function.

        :param i: row number with respect to predictor height
        :param j: column number with respect to predictor width
        """
        d_rows = abs(i - math.floor(kernel_size / 2))
        d_cols = abs(j - math.floor(kernel_size / 2))
        d = sqrt(d_rows**2 + d_cols**2)

        if d == 0:
            return 0.0
        else:
            return 1 / (1 + d / 2) ** 3

    vec_filter_func = np.vectorize(_filter_func)
    kernel_size = 2 * radius - 1
    shape = (kernel_size, kernel_size)
    ntl_filter = np.fromfunction(vec_filter_func, shape, dtype=float)
    filter_weights = ntl_filter / ntl_filter.sum()

    input_data_weights = np.zeros(shape=filter_weights.shape)
    input_data_weights[radius - 1, radius - 1] = 1.0

    return input_data_weights - filter_weights


class ElectrificationFilter(metaclass=abc.ABCMeta):
    """Abstract class for electrification filters. Electrification filters
    are filters that extract electrification targets from satellite imagery."""

    @abc.abstractmethod
    def predict(self, data: np.ndarray):
        pass


class NightlightFilter(ElectrificationFilter):
    """The electrification_predictor that was used in this Nature paper:
    https://www.nature.com/articles/s41597-019-0347-4"""

    def __init__(self, radius=21):
        """
        :param shape: Radius of square filter
        :type shape: int
        """
        self.radius = radius
        self.predictor = default_nightlight_target_filter(self.radius)

    def predict(self, data: np.ndarray):
        """
        Apply predictor on input data.
        :param data:
        :return:
        """
        prediction = signal.convolve2d(
            data, self.predictor, mode="same", boundary="symm"
        )
        return prediction


class NightlightTorchFilter(Conv2dImageFilter):
    """
    Implementation of NightlightFilter in PyTorch.
    """

    def __init__(self, radius=21, bias=False, threshold=0.4):
        super().__init__(radius=radius, bias=bias, padding_mode="reflect")
        self.threshold = threshold
        filter_weights = default_nightlight_target_filter(self.radius)
        self._set_weight(filter_weights)
        self.conv2.bias = nn.Parameter(torch.Tensor([threshold]).to(self.device))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        prediction = self.conv2(x)
        return prediction.squeeze()
