import geopandas as gpd
import rasterio
from gridfinder.metrics import eval_metrics
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)
import logging
from config import get_config

c = get_config(reload=True)

aoi_in = c.datafile_path("nigeria-boundary.geojson", stage=c.GROUND_TRUTH)
grid_truth = c.datafile_path("nigeriafinal.geojson", stage=c.GROUND_TRUTH)


guess_out = c.datafile_path(
    "nigeria_guess.tif", stage=c.PROCESSED, check_existence=False
)
log = c.log
logging.basicConfig(level=logging.INFO)

CELL_SIZES = [None, 10000, 15000]

df = gpd.read_file(grid_truth)
prediction = rasterio.open(guess_out)
aoi = gpd.read_file(aoi_in)

for size in CELL_SIZES:
    if size:
        logging.info("Evaluating with cell size: ")
        logging.info(size)
    log.info(grid_truth)
    log.info(aoi_in)
    log.info(guess_out)
    results = eval_metrics(
        df,
        prediction,
        cell_size_in_meters=size,
        aoi=aoi,
        metrics=[confusion_matrix, balanced_accuracy_score, classification_report],
    )
    for key, val in results.items():
        print(key, val)
