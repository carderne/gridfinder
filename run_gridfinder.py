#!/usr/bin/env python
# coding: utf-8
import logging
import os
import geopandas as gpd
import click

from src.gridfinder.gridfinder import optimise, estimate_mem_use, get_targets_costs
from src.gridfinder.post import raster_to_lines, thin, threshold_distances, accuracy
from src.gridfinder.util.raster import save_2d_array_as_raster, get_clipped_data
from src.gridfinder.util.loading import open_raster_in_tar
from src.gridfinder.prepare import (
    merge_rasters,
    drop_zero_pop,
    prepare_ntl,
    prepare_roads,
)
from gridfinder.electrificationfilter import NightlightFilter
from config import get_config
from data_access.remote_storage import RemoteStorage

from trains import Task, backend_api


@click.command()
@click.option(
    "--area-of-interest-data",
    default="nigeria/nigeria-kano.geojson",
    help="Path to AOI vector file relative to data/ground_truth/ directory.",
)
@click.option(
    "--roads-data",
    default="nigeria/nigeria-roads-200101.gpkg",
    help="Path to Roads vector file, relative to data/ground_truth/ directory",
)
@click.option(
    "--population-data",
    default="nigeria/population_nga_2018-10-01.tif",
    help="Path to population raster file, relative to data/ground_truth/ directory",
)
@click.option(
    "--grid-truth-data",
    default="nigeria/nigeriafinal.geojson",
    help="Path to ground truth grid vector file, relative to data/ground_truth/ directory",
)
@click.option(
    "--nightlight-data",
    default="nightlight_imagery/75N060W",
    help="Path to directory where nightlight imagery should be stored to, relative to data/raw/ directory",
)
@click.option(
    "--power-data",
    help="Path to vector file containing power lines, relative to data/processed/ directory",
)
@click.option(
    "--nightlight-output",
    default="nigeria/ntl_nigeria_clipped",
    help="Path to directory where clipped nightlight imagery will be stored,"
    "relative to data/processed directory.",
)
def run_gridfinder(
    area_of_interest_data: str,
    roads_data: str,
    population_data: str,
    grid_truth_data: str,
    nightlight_data: str,
    power_data: str,
    nightlight_output: str,
):
    DEFAULT_CRS = "EPSG:4326"
    c = get_config(reload=True)
    log = logging.getLogger(__name__)
    input_files = {}
    remote_storage = RemoteStorage(c.remote_storage)
    input_files["folder_ntl_in"] = c.datafile_path(
        nightlight_data, stage=c.RAW, check_existence=False, relative=True
    )
    input_files["aoi_in"] = c.datafile_path(
        area_of_interest_data,
        stage=c.GROUND_TRUTH,
        check_existence=False,
        relative=True,
    )
    input_files["roads_in"] = c.datafile_path(
        roads_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )
    input_files["pop_in"] = c.datafile_path(
        population_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )
    input_files["grid_truth"] = c.datafile_path(
        grid_truth_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )
    if power_data is not None:
        input_files["power"] = c.datafile_path(
            power_data, stage=c.PROCESSED, check_existence=False, relative=True
        )
    for _, path in input_files.items():
        remote_storage.pull(path, "")
    folder_ntl_out = c.datafile_path(
        nightlight_output, stage=c.PROCESSED, check_existence=False
    )
    raster_merged_out = c.datafile_path(
        "ntl_merged.tif", stage=c.PROCESSED, check_existence=False
    )
    targets_out = c.datafile_path(
        "targets.tif", stage=c.PROCESSED, check_existence=False
    )
    targets_clean_out = c.datafile_path(
        "targets_clean.tif", stage=c.CLEANED, check_existence=False
    )
    roads_out = c.datafile_path("roads.tif", stage=c.PROCESSED, check_existence=False)

    dist_out = c.datafile_path(
        "nigeria_dist.tif", stage=c.PROCESSED, check_existence=False
    )
    guess_out = c.datafile_path("guess.tif", stage=c.PROCESSED, check_existence=False)
    guess_skeletonized_out = c.datafile_path(
        "guess_skel.tif", stage=c.PROCESSED, check_existence=False
    )
    guess_vec_out = c.datafile_path(
        "guess.gpkg", stage=c.PROCESSED, check_existence=False
    )
    animate_out = os.path.join(c.visualizations, "guess.tif")

    params = {
        "percentile": 70,
        "ntl_threshold": 0.1,
        "upsample_by": 2,
        "cutoff": 0.0,
    }

    cfg = backend_api.load_config().to_dict()
    if not cfg["api"]["api_server"]:
        raise RuntimeError(
            "Prevented Trains experiment upload to AllegroAI demo server. Did you run 'trains init'?"
        )

    task = Task.init(
        project_name="Gridfinder",
        task_name="Nigeria Gridfinder run_gridfinder.py",
        reuse_last_task_id=True,
    )

    task.connect(input_files)
    task.connect(params)

    ntl_files_basedir = input_files["folder_ntl_in"]
    aoi = gpd.read_file(input_files["aoi_in"])

    for ntl_file in os.listdir(ntl_files_basedir):
        full_path = os.path.join(ntl_files_basedir, ntl_file)
        output_path = os.path.join(
            folder_ntl_out, f"{ntl_file[:-4]}.tif"
        )  # stripping off the .tgz
        with open_raster_in_tar(full_path, file_index=1) as raster:
            clipped_data, transform = get_clipped_data(raster, aoi)
            save_2d_array_as_raster(
                output_path, clipped_data, transform, crs=raster.crs.to_string()
            )
            log.info(f"Stored {full_path} as 2d raster in {output_path}.")

    raster_merged, affine, _ = merge_rasters(
        folder_ntl_out, percentile=params["percentile"]
    )
    save_2d_array_as_raster(raster_merged_out, raster_merged, affine, DEFAULT_CRS)
    log.info(
        f"Merged {len(raster_merged)} rasters of nightlight imagery to {raster_merged_out}"
    )

    ntl_filter = NightlightFilter()

    ntl_thresh, affine = prepare_ntl(
        raster_merged,
        affine,
        electrification_predictor=ntl_filter,
        threshold=params["ntl_threshold"],
        upsample_by=params["upsample_by"],
    )

    save_2d_array_as_raster(targets_out, ntl_thresh, affine, DEFAULT_CRS)
    log.info(f"Targets prepared from nightlight imagery and saved to {targets_out}.")

    targets_clean = drop_zero_pop(targets_out, input_files["pop_in"], aoi)
    save_2d_array_as_raster(targets_clean_out, targets_clean, affine, DEFAULT_CRS)
    log.info(f"Removed locations with zero population and saved to {targets_clean_out}")
    if power_data is not None:
        power = gpd.read_file(input_files["power"])
        roads_raster, affine = prepare_roads(
            input_files["roads_in"], aoi, targets_out, power
        )
    else:
        roads_raster, affine = prepare_roads(input_files["roads_in"], aoi, targets_out)
    save_2d_array_as_raster(roads_out, roads_raster, affine, DEFAULT_CRS, nodata=-1)
    log.info(
        f"Costs prepared and saved to {roads_out}, now connecting locations with algorithm."
    )

    targets, costs, start, affine = get_targets_costs(targets_clean_out, roads_out)
    est_mem = estimate_mem_use(targets, costs)
    log.info(f"Estimated memory usage of algorithm: {est_mem:.2f} GB")

    dist = optimise(
        targets,
        costs,
        start,
        jupyter=False,
        animate=False,
        affine=affine,
        animate_path=animate_out,
    )
    save_2d_array_as_raster(dist_out, dist, affine, DEFAULT_CRS)

    guess = threshold_distances(dist, threshold=params["cutoff"])
    save_2d_array_as_raster(guess_out, guess, affine, DEFAULT_CRS)
    log.info(f"Prediction is completed and saved to {guess_out}, thinning raster now..")

    task.upload_artifact("Prediction", guess)

    task.upload_artifact("Prediction raster", guess_out)

    guess_skel = thin(guess)
    save_2d_array_as_raster(guess_skeletonized_out, guess_skel, affine, DEFAULT_CRS)
    log.info(
        f"Thinning raster complete and saved to {guess_skeletonized_out}, now converting to vector geometries.."
    )

    guess_gdf = raster_to_lines(guess_skel, affine, DEFAULT_CRS)
    if power_data is not None:
        guess_gdf.append(power)
    guess_gdf.to_file(guess_vec_out, driver="GPKG")
    log.info(
        f"Converted raster to {len(guess_gdf)} grid lines and saved to "
        f"{guess_vec_out}. Evaluating on ground truth now.."
    )

    task.upload_artifact("Gridline Output", guess_vec_out)

    truth = gpd.read_file(input_files["grid_truth"]).to_crs(DEFAULT_CRS)
    true_pos, false_neg = accuracy(truth, guess_out, aoi)
    log.info(f"Points identified as grid that are grid: {100*true_pos:.0f}%")
    log.info(f"Actual grid that was missed: {100*false_neg:.0f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gridfinder()
