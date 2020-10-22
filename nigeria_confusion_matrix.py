import sys, os

# in order to get the config, it is not part of the library
# os.chdir("..")
# sys.path.append(os.path.abspath("."))


# In[2]:


import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import cm
import seaborn as sns
import rasterio
from gridfinder.metrics import eval_confusion_matrix
from config import get_config

c = get_config(reload=True)
folder_ntl_in = c.datafile_path("nightlight_imagery/75N060W", stage=c.RAW)
aoi_in = c.datafile_path("nigeria-boundary.geojson", stage=c.GROUND_TRUTH)
roads_in = c.datafile_path("nigeria-roads-200101.gpkg", stage=c.GROUND_TRUTH)
pop_in = c.datafile_path("population_nga_2018-10-01.tif", stage=c.GROUND_TRUTH)
grid_truth = c.datafile_path("nigeriafinal.geojson", stage=c.GROUND_TRUTH)

folder_ntl_out = c.datafile_path(
    "ntl_nigeria_clipped", stage=c.PROCESSED, check_existence=False
)
raster_merged_out = c.datafile_path(
    "ntl_nigeria_merged.tif", stage=c.PROCESSED, check_existence=False
)
targets_out = c.datafile_path(
    "nigeria_targets.tif", stage=c.PROCESSED, check_existence=False
)
targets_clean_out = c.datafile_path(
    "nigeria_targets_clean.tif", stage=c.CLEANED, check_existence=False
)
roads_out = c.datafile_path(
    "nigeria_roads.tif", stage=c.PROCESSED, check_existence=False
)

dist_out = c.datafile_path("nigeria_dist.tif", stage=c.PROCESSED, check_existence=False)
guess_out = c.datafile_path(
    "nigeria_guess_chris.tif", stage=c.PROCESSED, check_existence=False
)
guess_skeletonized_out = c.datafile_path(
    "nigeria_guess_skel.tif", stage=c.PROCESSED, check_existence=False
)
guess_nulled = c.datafile_path(
    "nigeria_guess_nulled.tif", stage=c.PROCESSED, check_existence=False
)
guess_vec_out = c.datafile_path(
    "nigeria_guess.gpkg", stage=c.PROCESSED, check_existence=False
)
animate_out = os.path.join(c.visualizations, "nigeria_guess_viz.tif")


df = gpd.read_file(grid_truth)
prediction = rasterio.open(guess_out)  #
aoi = gpd.read_file(aoi_in)
print(grid_truth, guess_out, aoi_in)
print("Cell size at raster resolution (ca. 220m)")
confusion = eval_confusion_matrix(df, prediction, aoi=aoi)
print(confusion)
accuracy = (confusion.tp + confusion.tn) / (
    confusion.tp + confusion.tn + confusion.fp + confusion.fn
)
precision = (confusion.tp) / (confusion.tp + confusion.fp)
recall = (confusion.tp) / (confusion.tp + confusion.fn)
print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)
print("New cell size 15km")
confusion = eval_confusion_matrix(df, prediction, cell_size_in_meters=15000, aoi=aoi)
print(confusion)
accuracy = (confusion.tp + confusion.tn) / (
    confusion.tp + confusion.tn + confusion.fp + confusion.fn
)
precision = (confusion.tp) / (confusion.tp + confusion.fp)
recall = (confusion.tp) / (confusion.tp + confusion.fn)
print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)

print("New cell size 10km")
confusion = eval_confusion_matrix(df, prediction, cell_size_in_meters=10000, aoi=aoi)
print(confusion)
accuracy = (confusion.tp + confusion.tn) / (
    confusion.tp + confusion.tn + confusion.fp + confusion.fn
)
precision = (confusion.tp) / (confusion.tp + confusion.fp)
recall = (confusion.tp) / (confusion.tp + confusion.fn)
print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)
