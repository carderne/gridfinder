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
from gridfinder.prepare import clip_rasters, merge_rasters, create_filter, prepare_ntl, prepare_roads
from gridfinder.gridfinder import get_targets_costs, estimate_mem_use, optimise
from gridfinder.post import threshold, accuracy, guess2geom

DEFAULT_PERCENTILE=70
DEFAULT_UPSAMPLE=2
DEAFULT_THRESHOLD=2.1
DEFAULT_CUTOFF=0.5

def main(country,
         percentile=DEFAULT_PERCENTILE,
         upsample=DEFAULT_UPSAMPLE,
         ntl_threshold=DEAFULT_THRESHOLD,
         cutoff=DEFAULT_CUTOFF):

    print(' - Running with')
    print('Country:', country)
    print('Percentile:', percentile)
    print('Upsample:', upsample)
    print('Threshold:', ntl_threshold)
    print('Cutoff:', cutoff)
    print()

    # Set file paths and clip AOI
    ntl_folder_in = Path('NTL')
    roads_in = Path('roads') / f'{country.lower()}.gpkg'

    aoi_in = 'gadm.gpkg'
    aoi = gpd.read_file(aoi_in)
    aoi = aoi.loc[aoi['NAME_0'] == country]

    folder_out = Path(country)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    ntl_folder_out = folder_out / 'NTL_clipped'
    ntl_merged_out = folder_out / 'ntl_merged.tif'
    ntl_thresh_out = folder_out / 'ntl_thresh.tif'
    roads_out = folder_out / 'roads.tif'
    dist_out = folder_out / 'dist.tif'
    guess_out = folder_out / 'guess.tif'
    final_out = folder_out / 'guess.gpkg'
    print(' - Done setup')

    # Clip NTL rasters and calculate nth percentile values
    clip_rasters(ntl_folder_in, ntl_folder_out, aoi)
    raster_merged, affine = merge_rasters(ntl_folder_out, percentile=percentile)
    save_raster(ntl_merged_out, raster_merged, affine)
    print(' - Done NTL percentile')

    # Apply filter to NTL
    ntl_filter = create_filter()
    ntl, ntl_filtered, ntl_interp, ntl_thresh, affine = prepare_ntl(ntl_merged_out, aoi, ntl_filter=ntl_filter,
                                                                    threshold=ntl_threshold, upsample_by=upsample)
    save_raster(ntl_thresh_out, ntl_thresh, affine)
    print(' - Done filter')

    # Create roads raster
    roads, roads_clipped, aoi, roads_raster, affine = prepare_roads(
        roads_in, aoi, ntl_thresh.shape, affine)
    save_raster(roads_out, roads_raster, affine)
    print(' - Done roads')

    # Load targets/costs and find a start point
    targets, costs, start, affine = get_targets_costs(ntl_thresh_out, roads_out)
    est_mem = estimate_mem_use(targets, costs)
    print(f'Estimated memory usage: {est_mem:.2f} GB')
    print(' - Done start point')

    # Run optimisation
    dist = optimise(targets, costs, start)
    save_raster(dist_out, dist, affine)
    print(' - Done optimisation')

    # Threshold optimisation output
    dists_r, guess, affine = threshold(dist_out, cutoff=cutoff)
    save_raster(guess_out, guess, affine)
    print(' - Done threshold')

    # Polygonize
    guess_r, guess_geojson, guess_gdf = guess2geom(guess_out)
    guess_gdf.to_file(final_out, driver='GPKG')
    print(' - Done polygonize')

    # Zip for download
    zip_file = Path(f'Downloads/{country}_{percentile}_{upsample}_{threshold}_{cutoff}')
    shutil.make_archive(zip_file, 'zip', folder_out)
    print(' - Done zip')


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--country', help='Country name, list in countries.csv', type=str)
    parser.add_argument('--percentile', help='Percentile cutoff for NTL layers', type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument('--upsample', help='Factor to upsample NTL resolution', type=int, default=DEFAULT_UPSAMPLE)
    parser.add_argument('--threshold', help='NTL values above this are considered electrified', type=float, default=DEAFULT_THRESHOLD)
    parser.add_argument('--cutoff', help='After the model run, dist values below this are considered grid', type=float, default=DEFAULT_CUTOFF)

    args=parser.parse_args()

    start = timer()

    main(country=args.country,
         percentile=args.percentile,
         upsample=args.upsample,
         ntl_threshold=args.threshold,
         cutoff=args.cutoff)

    end = timer()
    elapsed = (end - start) / 60 # to get to minutes
    print('Time taken:', elapsed, 'minutes') # Time in seconds, e.g. 5.38091952400282
