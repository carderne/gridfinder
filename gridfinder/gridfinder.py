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


def fix_shapes(targets, costs):
    """
    optimise won't work if the shapes aren't identical
    ACTUALLY I THINK IT MIGHT WHO KNOWS...
    """

    if targets.shape == costs.shape:
        return targets, costs
    
    else:
        # figure out how shape is different
        # use np.delete to get them the same?
        # carr = np.delete(carr, [0, 1], 0) # delete first two rows
        # carr = np.delete(carr, [0, 1], 1) # delete first two columns
        pass

def optimise(targets, costs, start, display_progress=False):
    """

    """
    
    counter = 0
    progress = 0
    max_cells = targets.shape[0] * targets.shape[1]
    
    max_i = costs.shape[0]
    max_j = costs.shape[1]    
    
    visited = np.zeros_like(costs)
    dist = np.full_like(costs, np.nan)
    prev = np.full_like(costs, np.nan, dtype=object)

    dist[start] = 0
    
    #       dist, loc
    queue = [[0, start]]
    heapify(queue)
    
    def zero_and_heap_path(loc):
        if not dist[loc] == 0:
            dist[loc] = 0
            visited[loc] = 1

            heappush(queue, [0, loc])
            prev_loc = prev[loc]

            if type(prev_loc) == tuple:
                zero_and_heap_path(prev_loc)
    
    if display_progress:
        handle = display(Markdown(''), display_id=True)
    while len(queue):
        current = heappop(queue)
        current_loc = current[1]
        current_i = current_loc[0]
        current_j = current_loc[1]
        current_dist = dist[current_loc]
        
        #print('\nCURRENT', current, 'DIST', current_dist, 'LEN', len(queue))
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
                
                # if the location is connected
                if targets[next_loc]:
                    prev[next_loc] = current_loc
                    zero_and_heap_path(next_loc)
                    #print('FOUND CONNECTED at', next_loc)
                
                # otherwise it's a normal queue cell
                else:
                    dist_add = costs[next_loc]
                    if x == 0 or y == 0: # if this cell is a square up/down or left/right
                        dist_add *= 1
                    else: # or if it's diagonal
                        dist_add *= sqrt(2)

                    next_dist = current_dist + dist_add

                    if visited[next_loc]:
                        if next_dist + 0.4 < dist[next_loc]:
                            #print('REVISITING at', next_loc, '  NEW DIST', next_dist)
                            dist[next_loc] = next_dist
                            prev[next_loc] = current_loc
                            heappush(queue, [next_dist, next_loc])

                    else:
                        #print('NEW CELL at', next_loc, '  DIST', next_dist)
                        heappush(queue, [next_dist, next_loc])
                        visited[next_loc] = 1
                        dist[next_loc] = next_dist
                        prev[next_loc] = current_loc

                        progress_new = 100 * counter/max_cells
                        if int(progress_new) > int(progress):
                            progress = progress_new
                            print(progress)
                            if display_progress:
                                handle.update(f'{progress:.2f}%')
                    
    return dist
