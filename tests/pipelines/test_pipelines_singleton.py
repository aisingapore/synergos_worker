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
from rest_rpc.core.pipelines.dataset import Singleton

##################
# Configurations #
##################

test_data_dir = os.path.join(app.config['TEST_DIR'], "pipelines", "data")
path_to_features = os.path.join(test_data_dir, "clean_engineered_headers.txt")
path_to_values = os.path.join(test_data_dir, "clean_engineered_data.npy")
path_to_types = os.path.join(test_data_dir, "clean_engineered_schema.json")
path_to_df = os.path.join(test_data_dir, "clean_combined_dataframe.csv")

df_schema = {
    "age": "int32", 
    "sex": "category", 
    "cp": "category", 
    "trestbps": "int32", 
    "chol": "int32", 
    "fbs": "category", 
    "restecg": "category", 
    "thalach": "int32", 
    "exang": "category", 
    "oldpeak": "float64", 
    "slope": "category", 
    "ca": "category", 
    "thal": "category", 
    "target": "category"
}

singleton = Singleton(
    features=path_to_features, 
    source=path_to_values, 
    types=path_to_types
)

#########################
# Singleton Class Tests #
#########################

def test_Singleton_header():
    with open(path_to_features, 'r') as fn:
        header = json.load(fn)
    assert header == singleton.header


def test_Singleton_schema():
    with open(path_to_types, 'r') as dt:
        schema = json.load(dt)
    assert schema == singleton.schema


def test_Singleton_values():
    values = np.load(path_to_values, allow_pickle=True)
    assert (values == singleton.values).all()


def test_Singleton_data():
    # Singletons should be agnostic to the data formatting applied before 
    # export. All formatting should be done before files are being exported.
    data = pd.read_csv(path_to_df, index_col=0)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    formatted_data = data.astype(dtype=df_schema)

    # Compare values only
    assert formatted_data.equals(singleton.data)
    # Compare schemas
    assert formatted_data.dtypes.to_dict() == singleton.data.dtypes.to_dict() 
