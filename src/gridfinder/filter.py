from math import sqrt
import abc
import numpy as np
from scipy import signal


class Filter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, data):
        pass


class NightlightFilter(Filter):
    def __init__(self, shape=(41, 41)):
        """Create and return a numpy array filter to be applied to the raster."""
        vec_filter_func = np.vectorize(self._filter_func)
        ntl_filter = np.fromfunction(vec_filter_func, shape, dtype=float)
        self.filter = ntl_filter / ntl_filter.sum()

    @staticmethod
    def _filter_func(i, j):
        d_rows = abs(i - 20)
        d_cols = abs(j - 20)
        d = sqrt(d_rows ** 2 + d_cols ** 2)

        if d == 0:
            return 0.0
        else:
            return 1 / (1 + d / 2) ** 3

    def predict(self, data):
        ntl_convolved = signal.convolve2d(data, self.filter, mode="same")
        ntl_filtered = data - ntl_convolved
        return ntl_filtered
