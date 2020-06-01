#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import json
import logging
import os
import random
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path

# Libs
import numpy as np
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

def seed_everything(seed=42):
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

def detect_configurations(dirname):
    """ Automates loading of configuration files in specified directory

    Args:
        dirname (str): Target directory to load configurations from
    Returns:
        Params (dict)
    """

    def parse_filename(filepath):
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

# State test directory
TEST_DIR = os.path.join(SRC_DIR, "tests")

# Initialise Cache
CACHE = infinite_nested_dict()

logging.debug(f"Is master node? {IS_MASTER}")
logging.debug(f"Input directory detected: {IN_DIR}")
logging.debug(f"Output directory detected: {OUT_DIR}")
logging.debug(f"Data directory detected: {DATA_DIR}")
logging.debug(f"Test directory detected: {TEST_DIR}")
logging.debug(f"Cache initialised: {CACHE}")

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
