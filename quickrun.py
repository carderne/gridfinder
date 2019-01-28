# gridfinder
# Licensed under GPL-3.0
# (c) Christopher Arderne

"""A simple command line script to run the full gridfinder package.
It uses quite a specific folder structure and might break with different files.

Input GADM: ~/data/gadm.gpkg
Input GHS: ~/data/ghs.tif
Input NTL fodler: ~/data/ntl/
Input roads: ~/data/roads/<country>.gpkg

Output data: ~/output/<country>/
Zipped output: ~/download/<country>_<parameters>.zip
"""

import argparse
import sys
import os
from pathlib import Path
import shutil
from timeit import default_timer as timer

import numpy as np
import rasterio
import geopandas as gpd

from gridfinder._util import save_raster, clip_line_poly
from gridfinder.prepare import clip_rasters, merge_rasters, create_filter, prepare_ntl, prepare_roads, drop_zero_pop
from gridfinder.gridfinder import get_targets_costs, estimate_mem_use, optimise
from gridfinder.post import threshold, accuracy, guess2geom

DEFAULT_PERCENTILE=70
DEFAULT_UPSAMPLE=2
DEAFULT_THRESHOLD=0.1
DEFAULT_CUTOFF=0.5

def main(country,
         percentile=DEFAULT_PERCENTILE,
         upsample=DEFAULT_UPSAMPLE,
         ntl_threshold=DEAFULT_THRESHOLD,
         cutoff=DEFAULT_CUTOFF,
         skip_ntl=False,
         skip_roads=False,
         drop_sites=False):
    """
    Run through the entire gridfinder process, skipping some steps depending on
    input parameters.
    """

    print(' - Running with:')
    print('Country:', country)
    print('Percentile:', percentile)
    print('Upsample:', upsample)
    print('Threshold:', ntl_threshold)
    print('Cutoff:', cutoff)
    print()
    print('Skip NTL:', skip_ntl)
    print('Skip roads:', skip_roads)
    print('Drop zero pop:', drop_sites)
    print()

    # Set file paths and clip AOI
    data_path = Path.home() / 'data'
    output_path = Path.home() / 'output'
    download_path = Path.home() / 'download'

    ntl_folder_in = data_path / 'ntl'
    pop_in = data_path / 'ghs.tif'
    roads_in = data_path / 'roads' / f'{country.lower()}.gpkg'
    aoi_in = data_path / 'gadm.gpkg'

    aoi = gpd.read_file(aoi_in)
    aoi = aoi.loc[aoi['NAME_0'] == country]

    folder_out = output_path / country
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    ntl_folder_out = folder_out / 'ntl_clipped'
    ntl_merged_out = folder_out / 'ntl_merged.tif'
    ntl_thresh_out = folder_out / 'ntl_thresh.tif'
    targets_out = folder_out / 'targets.tif'
    targets_out_clean = folder_out / 'targets_clean.tif'
    roads_out = folder_out / 'roads.tif'
    dist_out = folder_out / 'dist.tif'
    guess_out = folder_out / 'guess.tif'
    final_out = folder_out / 'guess.gpkg'

    zip_file = download_path / f'{country}_p{percentile}_u{upsample}_t{ntl_threshold}_c{cutoff}_d{drop_sites}'
    print(' - Done setup')

    def prep_ntl():
        # Clip NTL rasters and calculate nth percentile values
        clip_rasters(ntl_folder_in, ntl_folder_out, aoi)
        raster_merged, affine = merge_rasters(ntl_folder_out, percentile=percentile)
        save_raster(ntl_merged_out, raster_merged, affine)
        print(' - Done NTL percentile')

        # Apply filter to NTL
        ntl_filter = create_filter()
        ntl_thresh, affine = prepare_ntl(ntl_merged_out, aoi, ntl_filter=ntl_filter,
                                                                        threshold=ntl_threshold,
                                                                        upsample_by=upsample)
        save_raster(ntl_thresh_out, ntl_thresh, affine)
        print(' - Done filter')

        if drop_sites:
            print(' - Droping zero population sites')
            targets_clean = drop_zero_pop(ntl_thresh_out, pop_in, aoi)
            save_raster(targets_out, targets_clean, affine)
            save_raster(targets_out_clean, targets_clean, affine)
        else:
            save_raster(targets_out, ntl_thresh, affine)

    if skip_ntl:
        print(' - Skipping NTL steps')
    else:
        prep_ntl()

    def prep_roads():
        # Create roads raster
        roads_raster, affine = prepare_roads(
            roads_in, aoi, targets_out)
        save_raster(roads_out, roads_raster, affine)
        print(' - Done roads')

    if skip_roads:
        print(' - Skipping roads steps')
    else:
        prep_roads()

    # Load targets/costs and find a start point
    targets, costs, start, affine = get_targets_costs(targets_out, roads_out)
    est_mem = estimate_mem_use(targets, costs)
    print(f'Estimated memory usage: {est_mem:.2f} GB')
    print(' - Done start point')

    # Run optimisation
    dist = optimise(targets, costs, start)
    save_raster(dist_out, dist, affine)
    print(' - Done optimisation')

    # Threshold optimisation output
    guess, affine = threshold(dist_out, cutoff=cutoff)
    save_raster(guess_out, guess, affine)
    print(' - Done threshold')

    # Polygonize
    guess_gdf = guess2geom(guess_out)
    guess_gdf.to_file(final_out, driver='GPKG')
    print(' - Done polygonize')

    # Zip for download
    shutil.make_archive(zip_file, 'zip', folder_out)
    print(' - Done zip')

    print(' - Done for:')
    print('Country:', country)
    print('Percentile:', percentile)
    print('Upsample:', upsample)
    print('Threshold:', ntl_threshold)
    print('Cutoff:', cutoff)
    print()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--country', help='Country name, list in countries.csv', type=str)
    parser.add_argument('--percentile', help='Percentile cutoff for NTL layers', type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument('--upsample', help='Factor to upsample NTL resolution', type=int, default=DEFAULT_UPSAMPLE)
    parser.add_argument('--threshold', help='NTL values above this are considered electrified', type=float, default=DEAFULT_THRESHOLD)
    parser.add_argument('--cutoff', help='After the model run, dist values below this are considered grid', type=float, default=DEFAULT_CUTOFF)

    parser.add_argument('--skip-ntl', help='Skip NTL steps', action='store_true')
    parser.add_argument('--skip-roads', help='Skip roads step', action='store_true')
    parser.add_argument('--drop-sites', help='Drop sites with no pop', action='store_true')

    args=parser.parse_args()

    start = timer()

    main(country=args.country,
         percentile=args.percentile,
         upsample=args.upsample,
         ntl_threshold=args.threshold,
         cutoff=args.cutoff,
         skip_ntl=args.skip_ntl,
         skip_roads=args.skip_roads,
         drop_sites=args.drop_sites)

    end = timer()
    elapsed = (end - start) / 60 # to get to minutes
    print('Finished for', args)
    print('Time taken:', elapsed, 'minutes') # Time in seconds, e.g. 5.38091952400282

    times = Path.home() / 'times.txt'
    with times.open(mode='a') as f:
        f.write(f'{args}\n{elapsed} minutes\n\n')
