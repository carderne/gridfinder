import sys
from math import sqrt
from pathlib import Path
import json
from heapq import heapify, heappush, heappop

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import numpy as np
from scipy import signal
from skimage.graph import route_through_array

import rasterio

from IPython.display import display, Markdown

def get_targets(targets_in):
    """

    """
    targets_ra = rasterio.open(targets_in)
    targets = targets_ra.read(1)
    transform = targets_ra.transform

    target_list = np.argwhere(targets == 1.)
    start = tuple(target_list[0].tolist())

    return targets, transform, start


def get_costs(costs_in):
    """

    """
    costs_ra = rasterio.open(costs_in)
    costs = costs_ra.read(1)

    return costs


def optimise_generator(targets, costs, start, display_progress=False, use_direction=True):
    """

    """
    
    counter = 0
    progress = 0
    max_cells = targets.shape[0] * targets.shape[1]
    
    max_i = costs.shape[0]
    max_j = costs.shape[1]    
    
    visited = np.zeros_like(costs)
    dist = np.full_like(costs, np.nan)
    if use_direction:
        prev = np.full_like(costs, np.nan)
    else:
        prev = np.full_like(costs, np.nan, dtype=object)

    dist[start] = 0
    
    #       dist, loc
    halo = [[0, start]]
    heapify(halo)
    
    def zero_and_heap_path(loc):
        if not dist[loc] == 0:
            dist[loc] = 0
            visited[loc] = 1
            heappush(halo, [0, loc])

            if use_direction:
                prev_loc = get_location(at=loc, direction=prev[loc])
            else:
                prev_loc = prev[loc]

            if type(prev_loc) == tuple:
                zero_and_heap_path(prev_loc)
    
    if display_progress:
        handle = display(Markdown(''), display_id=True)
    while len(halo):
        current = heappop(halo)       
        current_loc = current[1]
        current_i = current_loc[0]
        current_j = current_loc[1]
        current_dist = dist[current_loc]
        
        #print()
        #print('CURRENT', current, 'DIST', current_dist)
        
        for x in range(-1,2):
            for y in range(-1,2):
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

                if use_direction:
                    dir_prev = get_direction(at=next_loc, to=current_loc)
                
                # if the location is connected
                if targets[next_loc]:
                    if use_direction:
                        prev[next_loc] = dir_prev
                    else:
                        prev[next_loc] = current_loc
                    zero_and_heap_path(next_loc)
                    #print('FOUND CONNECTED at', next_loc)
                
                # otherwise it's a normal halo cell
                else:
                    dist_add = costs[next_loc]
                    if x == 0 or y == 0: # if this cell is a square up/down or left/right
                        dist_add *= 1
                    else: # or if it's diagonal
                        dist_add *= sqrt(2)

                    next_dist = current_dist + dist_add

                    if visited[next_loc]:
                        if next_dist < dist[next_loc]:
                            #print('REVISITING at', next_loc, '  NEW DIST', next_dist)
                            dist[next_loc] = next_dist
                            if use_direction:
                                prev[next_loc] = dir_prev
                            else:
                                prev[next_loc] = current_loc
                            heappush(halo, [next_dist, next_loc])

                    else:
                        #print('NEW CELL at', next_loc, '  DIST', next_dist)
                        counter += 1


                        progress_new = 100 * counter/max_cells
                        heappush(halo, [next_dist, next_loc])
                        visited[next_loc] = 1
                        dist[next_loc] = next_dist
                        if use_direction:
                            prev[next_loc] = dir_prev
                        else:
                            prev[next_loc] = current_loc

                        if int(progress_new) > int(progress):
                            progress = progress_new
                            if display_progress:
                                handle.update(f'{progress:.2f}%')

                            yield np.copy(dist)
                        
                        #if counter > 100000:
                        #    return dist
                    
    yield np.copy(dist)


def optimise(targets, costs, start, display_progress=False, use_direction=True):
    """

    """

    generator = optimise_generator(targets, costs, start, display_progress)

    dist = None
    for dist in generator:
        pass

    return dist


def get_direction(at=None, to=None):
    """
    Direction based on this: (H is where we're at)
    5  4  3
    6  H  2
    7  0  1
    """

    at_x = at[1]
    at_y = at[0]
    
    to_x = to[1]
    to_y = to[0]
    
    diff_x = to_x - at_x
    diff_y = to_y - at_y
    
    if diff_x == 0 and diff_y == 1:
        return 0
    elif diff_x == 1 and diff_y == 1:
        return 1
    elif diff_x == 1 and diff_y == 0:
        return 2
    elif diff_x == 1 and diff_y == -1:
        return 3
    elif diff_x == 0 and diff_y == -1:
        return 4
    elif diff_x == -1 and diff_y == -1:
        return 5
    elif diff_x == -1 and diff_y == 0:
        return 6
    elif diff_x == -1 and diff_y == 1:
        return 7


def get_location(at=None, direction=None):
    """
    Direction based on this: (H is where we're at)
    5  4  3
    6  H  2
    7  0  1
    """

    diff_x = None
    diff_y = None
    
    if direction == 0:
        diff_x = 0
        diff_y = 1
    elif direction == 1:
        diff_x = 1
        diff_y = 1
    elif direction == 2:
        diff_x = 1
        diff_y = 0
    elif direction == 3:
        diff_x = 1
        diff_y = -1
    elif direction == 4:
        diff_x = 0
        diff_y = -1
    elif direction == 5:
        diff_x = -1
        diff_y = -1
    elif direction == 6:
        diff_x = -1
        diff_y = 0
    elif direction == 7:
        diff_x = -1
        diff_y = 1
        
    return (at[0] + diff_y, at[1] + diff_x)