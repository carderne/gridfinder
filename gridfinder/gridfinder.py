"""
Implements Dijkstra's algorithm on a cost-array to create an MST.
"""

import os
import pickle
import sys
from heapq import heapify, heappop, heappush
from math import sqrt
from typing import Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from IPython.display import Markdown, display

from gridfinder.util import save_raster

sys.setrecursionlimit(100000)


def get_targets_costs(
    targets_in: str, costs_in: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Affine]:
    """Load the targets and costs arrays from the given file paths.

    Parameters
    ----------
    targets_in : Path for targets raster.
    costs_in : Path for costs raster.

    Returns
    -------
    targets : 2D array of targets
    costs: 2D array of costs
    start: tuple with row, col of starting point.
    affine : Affine transformation for the rasters.
    """

    targets_ra = rasterio.open(targets_in)
    affine = targets_ra.transform
    targets = targets_ra.read(1)

    costs_ra = rasterio.open(costs_in)
    costs = costs_ra.read(1)

    target_list = np.argwhere(targets == 1.0)
    start = tuple(target_list[0].tolist())

    targets = targets.astype(np.int8)
    costs = costs.astype(np.float16)

    return targets, costs, start, affine


def estimate_mem_use(targets: np.ndarray, costs: np.ndarray) -> float:
    """Estimate memory usage in GB, probably not very accurate.

    Parameters
    ----------
    targets : 2D array of targets.
    costs : 2D array of costs.

    Returns
    -------
    est_mem : Estimated memory requirement in GB.
    """

    # make sure these match the ones used in optimise below
    visited = np.zeros_like(targets, dtype=np.int8)
    dist = np.full_like(costs, np.nan, dtype=np.float32)
    prev = np.full_like(costs, np.nan, dtype=object)

    est_mem_arr = [targets, costs, visited, dist, prev]
    est_mem = len(pickle.dumps(est_mem_arr, -1))

    return est_mem / 1e9


def optimise(
    targets: np.ndarray,
    costs: np.ndarray,
    start: Tuple[int, int],
    jupyter: bool = False,
    animate: bool = False,
    affine: Optional[Affine] = None,
    animate_path: str = "",
    silent: bool = False,
) -> np.ndarray:
    """Run the Dijkstra algorithm for the supplied arrays.

    Parameters
    ----------
    targets : 2D array of targets.
    costs : 2D array of costs.
    start : tuple with row, col of starting point.
    jupyter : Whether the code is being run from a Jupyter Notebook.

    Returns
    -------
    dist :
        2D array with the distance (in cells) of each point from a 'found'
        on-grid point. Values of 0 imply that cell is part of an MV grid line.
    """


    max_i = costs.shape[0]
    max_j = costs.shape[1]

    visited = np.zeros_like(targets, dtype=np.int8)
    dist = np.full_like(costs, np.nan, dtype=np.float32)

    # want to set this to dtype='int32, int32'
    # but then the if type(prev_loc) == tuple check will break
    # becuas it gets instantiated with tuples
    prev = np.full_like(costs, np.nan, dtype=object)

    dist[start] = 0

    #       dist, loc
    queue = [[0, start]]
    heapify(queue)

    def zero_and_heap_path(loc: Tuple[int, int]) -> None:
        """Zero the location's distance value and follow upstream doing same.

        Parameters
        ----------
        loc : row, col of current point.
        """

        if not dist[loc] == 0:
            dist[loc] = 0
            visited[loc] = 1

            heappush(queue, [0, loc])
            prev_loc = prev[loc]

            if type(prev_loc) == tuple:
                zero_and_heap_path(prev_loc)

    counter = 0
    progress = 0
    max_cells = targets.shape[0] * targets.shape[1]
    handle = None
    if jupyter:
        handle = display(Markdown(""), display_id=True)

    while len(queue):
        current = heappop(queue)
        current_loc = current[1]
        current_i = current_loc[0]
        current_j = current_loc[1]
        current_dist = dist[current_loc]

        for x in range(-1, 2):
            for y in range(-1, 2):
                next_i = current_i + x
                next_j = current_j + y
                next_loc = (next_i, next_j)

                # ensure we're within bounds
                if next_i < 0 or next_j < 0 or next_i >= max_i or next_j >= max_j:
                    continue

                # ensure we're not looking at the same spot
                if next_loc == current_loc:
                    continue

                # skip if we've already set dist to 0
                if dist[next_loc] == 0:
                    continue

                # if the location is connected
                if targets[next_loc]:
                    prev[next_loc] = current_loc
                    zero_and_heap_path(next_loc)

                # otherwise it's a normal queue cell
                else:
                    dist_add = costs[next_loc]
                    if x == 0 or y == 0:  # if this cell is  up/down/left/right
                        dist_add *= 1
                    else:  # or if it's diagonal
                        dist_add *= sqrt(2)

                    next_dist = current_dist + dist_add

                    if visited[next_loc]:
                        if next_dist < dist[next_loc]:
                            dist[next_loc] = next_dist
                            prev[next_loc] = current_loc
                            heappush(queue, [next_dist, next_loc])

                    else:
                        heappush(queue, [next_dist, next_loc])
                        visited[next_loc] = 1
                        dist[next_loc] = next_dist
                        prev[next_loc] = current_loc

                        counter += 1
                        progress_new = 100 * counter / max_cells
                        if int(progress_new) > int(progress):
                            progress = progress_new
                            message = f"{progress:.2f} %"
                            if jupyter and handle:
                                handle.update(message)
                            elif not silent:
                                print(message)
                            if animate:
                                i = int(progress)
                                path = os.path.join(animate_path, f"arr{i:03d}.tif")
                                if not affine:
                                    raise ValueError("Must provide an affine when animate=True")  # NoQA
                                save_raster(path, dist, affine)

    return dist
