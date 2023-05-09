"""
This script evaluates the results of grid-finder analysis.
The experiment will be tracked in clearml.
"""
import logging
import os
import sys
from typing import Dict, List

import click
import geopandas as gpd
import pandas as pd
import rasterio
from clearml import Task
from gridfinder.metrics import get_binary_arrays
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.append(os.path.abspath("."))
from config import get_config
from util.clearml import assert_clearml_config_exists, download_clearml_artifact
from util.remote_storage import get_default_remote_storage, get_develop_remote_storage

log = logging.getLogger(__name__)
METRICS = [confusion_matrix, balanced_accuracy_score, classification_report]


@click.command()
@click.option(
    "--aoi-in", "-a", default="burundi/gadm.gpkg", help="Area for the evaluation."
)
@click.option(
    "--grid-truth-in", "-g", default="burundi/grid.gpkg", help="Ground truth grid."
)
@click.option(
    "--cell-sizes",
    "-c",
    multiple=True,
    default=[500, 1000, 2500, 5000],
    type=int,
    help="Cell sizes to run the evaluation on. Passed as multiple cmd args."
    "Example: python ... -c 100 -c 1000 -c 10000",
)
@click.option(
    "--prediction-file",
    "-p",
    default="burundi/guess.tif",
    help="Local path to the tif containing the prediction."
    "Output of the grid-finder Analysis.",
)
@click.option("--download-prediction-from-clearml", "-d", is_flag=True, default=False)
@click.option(
    "--clearml-project-name",
    default="Gridfinder",
    help="Name of the project for this clearml task. Useful for differentiating CI runs",
)
@click.option(
    "--prediction-task-name",
    default="Evaluation",
    help="The name of the clearml task which produced the prediction.",
)
@click.option(
    "--prediction-task-id",
    default="",
    help="Id of the clearml task that produced the prediction.",
)
@click.option(
    "--prediction-artifact-name",
    default="Prediction raster",
    help="Name of the prediction artifact.",
)
@click.option(
    "--prediction-artifact-file",
    default="guess.tif",
    help="Name of the file containing the prediction on the clearml server.",
)
@click.option(
    "--dev-mode",
    is_flag=True,
    help="Whether to run the script in dev mode. "
    "If true, the results will be pushed to the development database",
)
def run_evaluation(
    aoi_in: str,
    grid_truth_in: str,
    cell_sizes: List[int],
    clearml_project_name: str,
    prediction_file: str,
    download_prediction_from_clearml: bool,
    prediction_task_name: str,
    prediction_task_id: str,
    prediction_artifact_name: str,
    prediction_artifact_file: str,
    dev_mode: bool,
):
    # snapshot the function arguments
    parameters = click.get_current_context().params
    c = get_config(reload=True)

    if dev_mode:
        remote_storage = get_develop_remote_storage()
    else:
        remote_storage = get_default_remote_storage()

    aoi_in = c.datafile_path(
        aoi_in, stage=c.GROUND_TRUTH, relative=True, check_existence=False
    )
    grid_truth_in = c.datafile_path(
        grid_truth_in, stage=c.GROUND_TRUTH, relative=True, check_existence=False
    )
    if download_prediction_from_clearml:
        prediction_in = c.datafile_path(
            f"gridfinder/{prediction_artifact_name}/{prediction_task_id}/{prediction_artifact_file}",
            stage=c.PROCESSED,
            relative=True,
            check_existence=False,
        )
    else:
        prediction_in = c.datafile_path(
            prediction_file, stage=c.PROCESSED, relative=True, check_existence=False
        )

    log.info("Downloading files aoi and grid truth from remote storage.")
    remote_storage.pull(aoi_in)
    remote_storage.pull(grid_truth_in)
    if not download_prediction_from_clearml:
        remote_storage.pull(prediction_in)

    if download_prediction_from_clearml:
        log.info("Downloading prediction from clearml server.")
        download_clearml_artifact(
            project_name="Gridfinder",
            task_name=prediction_task_name,
            task_id=prediction_task_id,
            artifact_name=prediction_artifact_name,
            artifact_file=prediction_artifact_file,
            output_file=prediction_in,
        )

    log.info("Loading files into memory ...")
    aoi = gpd.read_file(aoi_in)
    grid_truth = gpd.read_file(grid_truth_in)
    prediction = rasterio.open(prediction_in)

    grid_truth = grid_truth.to_crs(aoi.crs)

    assert_clearml_config_exists()

    task = Task.init(
        project_name=clearml_project_name,
        task_name="Evaluation",
        reuse_last_task_id=False,
    )

    # save all parameters passed to this function
    task.connect(parameters)
    task.connect(
        {"Command": sys.executable.split("/")[-1] + " " + " ".join(sys.argv)},
        name="Full Command",
    )
    # set the id of the analysis for transparency
    task.set_tags([f"Analysis {prediction_task_name}/{prediction_task_id}"])

    report_output = c.artifact_path(
        name="gridfinder", relative=True, check_existence=False
    )
    os.makedirs(report_output, exist_ok=True)
    report_path = os.path.join(report_output, f"gridfinder_report.txt")
    _ = open(report_path, "w")  # empty possibly existing file
    for cell_size in cell_sizes:

        info_string = f"Evaluation {prediction_in} on cell size {cell_size}"
        log.info(info_string)

        y_pred, y_true = get_binary_arrays(
            grid_truth, prediction, cell_size_in_meters=cell_size, aoi=aoi
        )

        report_file_stream = open(report_path, "a")

        report_file_stream.write(info_string + "\n")
        report_file_stream.write("\n")
        for metric in METRICS:
            measure = metric(y_true=y_true, y_pred=y_pred)
            report_file_stream.write(metric.__name__ + "\n")
            report_file_stream.write(str(measure) + "\n")
            report_file_stream.write("\n")

        report_file_stream.write("-" * 30 + "\n" * 3)
        report_file_stream.close()

        task.logger.report_confusion_matrix(
            title=f"Confusion matrix for cell size: {cell_size}",
            series=str(cell_size),
            matrix=confusion_matrix(y_true=y_true, y_pred=y_pred),
            iteration=0,
        )
        task.logger.report_scalar(
            title="Balanced accuracy score",
            series=str(cell_size),
            value=balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
            iteration=0,
        )
        task.logger.report_table(
            title=f"Classification Report for cell size: {cell_size}",
            series=str(cell_size),
            iteration=0,
            table_plot=construct_dataframe(
                classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
            ),
        )

    task.upload_artifact(name="Report", artifact_object=report_path)


def construct_dataframe(report: Dict) -> pd.DataFrame:
    # remove scalar
    if report["accuracy"]:
        del report["accuracy"]
    return pd.DataFrame.from_dict(report, orient="index")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
