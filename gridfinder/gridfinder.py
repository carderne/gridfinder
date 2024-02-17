"""
Implements Dijkstra's algorithm on a cost-array to create an MST.
"""

import pickle
from heapq import heapify, heappop, heappush
from math import sqrt
from typing import List, Optional, Tuple

import numba as nb
import numpy as np
import rasterio
from affine import Affine

from gridfinder.util import Loc, Pathy


def get_targets_costs(
    targets_in: Pathy,
    costs_in: Pathy,
) -> Tuple[np.ndarray, np.ndarray, Loc, Affine]:
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
    costs = costs.astype(np.float32)

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


@nb.njit
def optimise(
    targets: np.ndarray,
    costs: np.ndarray,
    start: Loc,
    silent: bool = False,
    jupyter: bool = False,
    animate: bool = False,
    affine: Optional[Affine] = None,
    animate_path: str = "",
) -> np.ndarray:
    """Run the Dijkstra algorithm for the supplied arrays.

    Parameters
    ----------
    targets : 2D array of targets.
    costs : 2D array of costs.
    start : tuple with row, col of starting point.
    silent : whether to print progress

    Returns
    -------
    dist :
        2D array with the distance (in cells) of each point from a 'found'
        on-grid point. Values of 0 imply that cell is part of an MV grid line.
    """

    if jupyter or animate or affine or animate_path:
        print(
            "Warning: the following parameters are ignored: jupyter, animate, affine, animate_path"  # NoQA
        )

    shape = costs.shape
    max_i = shape[0]
    max_j = shape[1]

    visited = np.zeros(shape, dtype=np.int8)
    dist = np.full(shape, np.nan, dtype=np.float32)
    prev = np.full((shape[0], shape[1], 2), -1, dtype=np.int32)

    dist[start] = 0

    queue: List[Tuple[float, Loc]] = [(0.0, start)]
    heapify(queue)

    counter = 0
    progress = 0
    max_cells = targets.shape[0] * targets.shape[1]

    while len(queue):
        _, current_loc = heappop(queue)
        current_dist = dist[current_loc]

        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                next_i = current_loc[0] + x
                next_j = current_loc[1] + y
                next_loc = (next_i, next_j)

                if (
                    (x == 0 and y == 0)  # same spot
                    or dist[next_loc] == 0.0  # already zerod
                    or next_i < 0  # out of bounds
                    or next_j < 0
                    or next_i >= max_i
                    or next_j >= max_j
                ):
                    continue

                # if the location is connected
                if targets[next_loc]:
                    prev[next_loc] = current_loc
                    zero_loc = next_loc
                    while not dist[zero_loc] == 0.0:
                        dist[zero_loc] = 0.0
                        visited[zero_loc] = 1

                        heappush(queue, (0.0, zero_loc))
                        new_zero_loc = prev[zero_loc]
                        zero_loc = (new_zero_loc[0], new_zero_loc[1])
                        if zero_loc[0] == -1:
                            break

                # otherwise it's a normal queue cell
                else:
                    dist_add = costs[next_loc]
                    if x != 0 and y != 0:  # diagonal
                        dist_add *= sqrt(2)

                    next_dist = current_dist + dist_add

                    # visited before
                    if visited[next_loc]:
                        if next_dist < dist[next_loc]:
                            dist[next_loc] = next_dist
                            prev[next_loc] = current_loc
                            heappush(queue, (next_dist, next_loc))

                    # brand new cell - progress!
                    else:
                        heappush(queue, (next_dist, next_loc))
                        visited[next_loc] = 1
                        dist[next_loc] = next_dist
                        prev[next_loc] = current_loc

                        if not silent:
                            counter += 1
                            progress_new = int(100 * counter / max_cells)
                            if progress_new > progress:
                                progress = progress_new
                                with nb.objmode():
                                    print_progress(progress)
    print()
    return dist


def print_progress(progress: int) -> None:
    if progress % 5 == 0:
        print(progress, end="", flush=True)
    else:
        print(".", end="", flush=True)
