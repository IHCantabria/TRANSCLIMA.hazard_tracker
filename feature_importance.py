import json
import os
from tracker_utils import Random_forest_feature_importance

from pathlib import Path
import argparse
from datetime import datetime


## example of how to run this code 
# conda activate transcatalog
#python main.py config3.json --basin=NA --storm_index=0 

parser = argparse.ArgumentParser(description="Run surge processing pipeline with JSON config.")

parser.add_argument("config_file", help="Path to JSON configuration file")
parser.add_argument("--basin", type=str, help="Optional override for storm_index")

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

# Assign variables
basin           = config["basin"]
variable_name   = config["variable_name"]
case_id         = config["case_id"]

# Define path
folder = os.getcwd()
folder = Path(folder)

folder_out='storm_catalogue_ss_'+case_id

pathout0 = folder / f"{folder_out}/{basin}/"
pathout0.mkdir(parents=True, exist_ok=True)


# var can be 
#       maximum storm surge  = ssmax
#       storm surge duration = ssdur


Random_forest_feature_importance(basin=basin, var='ssmax', folder=folder,pathout=pathout0)



