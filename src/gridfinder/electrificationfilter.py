import math
from math import sqrt
import abc
import numpy as np
from scipy import signal


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
        :param shape: Shape of filter, e.g. (2,2)
        """
        self.shape = shape
        self.filter = self._func_to_filter()

    def _filter_func(self, i: int, j: int):
        """
        d is that pixel’s perpendicular distance from a given square sample’s centroid.
        The goal of this filter is to find pixels with a higher value than their neighborhood,
        biased towards closer areas. To achieve this bias, a non-linear function is needed,
        and a cubic function was found to achieve better results than a square function.

        :param i: row number with respect to filter height
        :param j: column number with respect to filter width
        :return: Value
        """
        d_rows = abs(i - math.floor(self.shape[0] / 2))
        d_cols = abs(j - math.floor(self.shape[1] / 2))
        d = sqrt(d_rows ** 2 + d_cols ** 2)

        if d == 0:
            return 0.0
        else:
            return 1 / (1 + d / 2) ** 3

    def _func_to_filter(self):
        """
        Construct a filter from the pixel-wise function.
        The filter is normalized so that and the output is then subtracted from the original image.
        :return: Numpy array with filter values
        """
        vec_filter_func = np.vectorize(self._filter_func)
        ntl_filter = np.fromfunction(vec_filter_func, self.shape, dtype=float)
        return ntl_filter / ntl_filter.sum()

    def predict(self, data: np.ndarray):
        """
        Apply filter on input data.
        :param data:
        :return:
        """
        ntl_convolved = signal.convolve2d(data, self.filter, mode="same")
        ntl_filtered = data - ntl_convolved
        return ntl_filtered
