#!/usr/bin/env python

"""
Remember to assign all configurable variables in CAPS (eg. OUT_DIR).
This is because Flask-restx will only load all uppercase variables from
`config.py`.
"""

####################
# Required Modules #
####################

# Generic
import json
import logging
import os
import random
import subprocess
import sys
import zipfile
from collections import defaultdict, OrderedDict
from glob import glob
from multiprocessing import Manager
from string import Template
from pathlib import Path
from typing import Dict

# Libs
import numpy as np
import psutil
import torch as th

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

SRC_DIR = Path(__file__).parent.absolute()

API_VERSION = "0.0.1"

####################
# Helper Functions #
####################

def seed_everything(seed: int = 42) -> bool:
    """ Convenience function to set a constant random seed for model consistency

    Args:
        seed (int): Seed for RNG
    Returns:
        True    if operation is successful
        False   otherwise
    """
    try:
        random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return True

    except:
        return False


def count_available_gpus() -> int:
    """ Counts no. of attached GPUs devices in the current system. As GPU 
        support is supplimentary, if any exceptions are caught here, system
        defaults back to CPU-driven processes (i.e. gpu count is 0)

    Returns:
        gpu_count (int)
    """
    try:
        process = subprocess.run(
            ['lspci'],
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        all_detected_devices = process.stdout.split('\n')
        gpus = [
            device 
            for device in all_detected_devices 
            if ('VGA' in device) or ('Display' in device)
        ]
        return len(gpus)

    except subprocess.CalledProcessError as cpe:
        logging.warning(f"Could not detect GPUs! Error: {cpe}")
        logging.warning(f"Defaulting to CPU processing instead...")
        return 0


def detect_configurations(dirname: str) -> Dict[str, str]:
    """ Automates loading of configuration files in specified directory

    Args:
        dirname (str): Target directory to load configurations from
    Returns:
        Params (dict)
    """

    def parse_filename(filepath: str) -> str:
        """ Extracts filename from a specified filepath
            Assumptions: There are no '.' in filename
        
        Args:
            filepath (str): Path of file to parse
        Returns:
            filename (str)
        """
        return os.path.basename(filepath).split('.')[0]

    # Load in parameters for participating servers
    config_globstring = os.path.join(SRC_DIR, dirname, "*.json")
    config_paths = glob(config_globstring)

    return {parse_filename(c_path): c_path for c_path in config_paths}


def install(package: str) -> bool:
    """ Allows for dynamic runtime installation of python modules. 
    
        IMPORTANT: 
        Modules specified will be installed from source, meaning that `package` 
        must be a path to some `.tar.gz` archive.

    Args:
        package (str): Path to distribution package for installation
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True

    except:
        return False

################################################
# PySyft Worker Container Local Configurations #
################################################
""" 
General parameters required for processing inputs & outputs
"""

# Define server's role: Master or slave
IS_MASTER = False

# State input directory
IN_DIR = os.path.join(SRC_DIR, "inputs")

# State output directory
OUT_DIR = os.path.join(SRC_DIR, "outputs")

# State data directory
DATA_DIR = os.path.join(SRC_DIR, "data")

# State model directory
MODEL_DIR = os.path.join(SRC_DIR, "models")

# State custom directory
CUSTOM_DIR = os.path.join(SRC_DIR, "custom")

# State test directory
TEST_DIR = os.path.join(SRC_DIR, "tests")

# Initialise Cache
CACHE = Manager().dict()#infinite_nested_dict()

# Allocate no. of cores for processes
CORES_USED = psutil.cpu_count(logical=True) - 1

# Detect no. of GPUs attached to server
GPU_COUNT = count_available_gpus()

logging.debug(f"Is master node? {IS_MASTER}")
logging.debug(f"Input directory detected: {IN_DIR}")
logging.debug(f"Output directory detected: {OUT_DIR}")
logging.debug(f"Data directory detected: {DATA_DIR}")
logging.debug(f"Test directory detected: {TEST_DIR}")
logging.debug(f"Cache initialised: {CACHE}")
logging.debug(f"No. of available CPU Cores: {CORES_USED}")
logging.debug(f"No. of available GPUs: {GPU_COUNT}")

#########################################
# PySyft Worker Database Configurations #
#########################################
""" 
In PySyft worker, the database is used mainly for caching results of operations
triggered by the TTP's REST-RPC calls
"""
DB_PATH = os.path.join(SRC_DIR, "outputs", "operations.json")

logging.debug(f"Database path detected: {DB_PATH}")

##################################
# PySyft Worker Template Schemas #
##################################
"""
For REST service to be stable, there must be schemas enforced to ensure that any
erroneous queries will affect the functions of the system.
"""
template_paths = detect_configurations("templates")

SCHEMAS = {}
for name, s_path in template_paths.items():
    with open(s_path, 'r') as schema:
        SCHEMAS[name] = json.load(schema, object_pairs_hook=OrderedDict)

logging.debug(f"Schemas loaded: {list(SCHEMAS.keys())}")

#######################################
# PySyft Worker Export Configurations #
####################################### 
"""
Certain Flask requests sent from the TTP (namely `/poll` and `/predict`) will
trigger file exports to the local machine, while other requests 
(i.e. `initialise`) perform lazy loading and require access to these exports.
This will ensure that all exported filenames are consistent during referencing.
"""
cache_dir = os.path.join(OUT_DIR, "$project_id", "preprocessing")
aggregated_X_outpath = os.path.join(cache_dir, "preprocessed_X_$meta.npy")
aggregated_y_outpath = os.path.join(cache_dir, "preprocessed_y_$meta.npy")
aggregated_df_outpath = os.path.join(cache_dir, "combined_dataframe_$meta.csv")
CACHE_TEMPLATE = {
    'out_dir': Template(cache_dir),
    'X': Template(aggregated_X_outpath),
    'y': Template(aggregated_y_outpath),
    'dataframe': Template(aggregated_df_outpath)
}

predict_outdir = os.path.join(
    OUT_DIR, 
    "$project_id", 
    "$expt_id", 
    "$run_id", 
    "$meta"
)
y_pred_outpath = os.path.join(predict_outdir, "inference_predictions_$meta.txt")
y_score_outpath = os.path.join(predict_outdir, "inference_scores_$meta.txt")
stats_outpath = os.path.join(predict_outdir, "inference_statistics_$meta.json")
PREDICT_TEMPLATE = {
    'out_dir': Template(predict_outdir),
    'y_pred': Template(y_pred_outpath),
    'y_score': Template(y_score_outpath),
    'statistics': Template(stats_outpath)
}

#######################################
# PySyft Flask Payload Configurations #
####################################### 
"""
Responses for REST-RPC have a specific format to allow compatibility between TTP
& Worker Flask Interfaces. Remember to modify rest_rpc.connection.core.utils.Payload 
upon modifying this template!
"""
PAYLOAD_TEMPLATE = {
    'apiVersion': API_VERSION,
    'success': 0,
    'status': None,
    'method': "",
    'params': {},
    'data': {}
}

############################
# REST-RPC Language Models #
############################

# Install language models for Spacy
spacy_src_dir = Path(os.path.join(CUSTOM_DIR, 'spacy'))
spacy_sources = list(spacy_src_dir.glob('**/*.tar.gz'))
for sp_src in spacy_sources:
    install(sp_src)

# Load all user-declared source paths for Symspell
symspell_src_dir = Path(os.path.join(CUSTOM_DIR, 'symspell'))
SYMSPELL_DICTIONARIES = list(symspell_src_dir.glob('**/*dictionary*.txt'))
SYMSPELL_BIGRAMS = list(symspell_src_dir.glob('**/*bigram*.txt'))

# Load all user-declared data paths for NLTK
nltk_src_dir = os.path.join(CUSTOM_DIR, 'nltk_data')
os.environ["NLTK_DATA"] = nltk_src_dir
nltk_sources = list(Path(nltk_src_dir).glob('**/*.zip'))
for nltk_src in nltk_sources:
    print(nltk_src)
    with zipfile.ZipFile(nltk_src,"r") as zip_ref:
        zip_ref.extractall(path=nltk_src.parent)
