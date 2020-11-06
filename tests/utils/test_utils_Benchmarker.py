#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import os

# Libs
import numpy as np
import pandas as pd

# Custom
from rest_rpc import app
from rest_rpc.core.utils import Benchmarker

##################
# Configurations #
##################

test_data_dir = os.path.join(app.config['TEST_DIR'], "utils", "data")
path_to_y_pred = os.path.join(test_data_dir, "inference_predictions.txt")
path_to_y_score = os.path.join(test_data_dir, "inference_scores.txt")


###########################
# Benchmarker Class Tests #
###########################
