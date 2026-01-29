import json
import os
from tracking_ss import tracking_ss,hydrograph,tracking_gesla,hydrograph_gesla

from pathlib import Path
import argparse
from datetime import datetime


## example of how to run this code 
# conda activate transcatalog
#python main.py config3.json --basin=NA --storm_index=0 



parser = argparse.ArgumentParser(description="Run surge processing pipeline with JSON config.")

parser.add_argument("config_file", help="Path to JSON configuration file")
parser.add_argument("--basin", type=str, help="Optional override for storm_index")
parser.add_argument("--storm_index", type=int, help="Optional override for storm_index")

args = parser.parse_args()

# ============================
# 2. Load configuration from JSON
# ============================
config_path = Path(args.config_file)
with open(config_path) as f:
    config = json.load(f)

# Override storm_index if provided
if args.basin is not None:
    config["basin"] = args.basin


if args.storm_index is not None:
    config["storm_index"] = args.storm_index    

# Assign variables
basin           = config["basin"]
storm_index     = config["storm_index"]
threshold_peak  = config["threshold_peak"]
threshold_min   = config["threshold_min"]
eps             = config["DBSCAN_eps"]
min_samples     = config["DBSCAN_min_samples"]
delta           = config["delta"]
variable_name   = config["variable_name"]
case_id         = config["case_id"]
KM_ncentroids   = config["KM_ncentroids"]

# Define path
folder = os.getcwd()
folder = Path(folder)

folder_out='storm_catalogue_ss_'+case_id

pathout0 = folder / f"{folder_out}/{basin}/"
pathout0.mkdir(parents=True, exist_ok=True)

print(basin)

# Run first script
tracking_ss(basin=basin, ii=storm_index, folder=folder, eps=eps, min_samples=min_samples, delta=delta, threshold_peak=threshold_peak,variable_name='ss',pathout=pathout0,ncentroids=KM_ncentroids)
hydrograph(basin, ii=storm_index, threshold=threshold_min, folder=folder,variable_name='ss',pathout=pathout0)

#tracking_gesla(basin=basin, ii=storm_index, folder=folder, eps=eps, min_samples=min_samples, delta=delta, threshold_peak=threshold_peak,variable_name='GESLA',pathout=pathout0,ncentroids=KM_ncentroids)
#hydrograph_gesla(basin, ii=storm_index, threshold=threshold_min, folder=folder,variable_name='GESLA',pathout=pathout0)





