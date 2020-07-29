#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import List

# Libs
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from tqdm import tqdm

# Custom
from rest_rpc.core.pipelines.base import BasePipe
from rest_rpc.core.pipelines.preprocessor import Preprocessor
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################


##########################################
# Data Preprocessing Class - TabularPipe #
##########################################

class TabularPipe(BasePipe):
    """
    The TabularPipe class prepares a specified dataset for model fitting by
    applying a generalised process of interpolation that is non-data-specific,
    before finally applying certain encryption/noising methods to introduce a 
    sufficient level of noise such that the identity of an individual becomes
    statistically untraceable.

    Prerequisite: Data MUST have its labels headered as 'target'

    Attributes:
        des_dir (str): Destination directory to save data in
        data   (list(str)): Loaded data to be processed
        output (PipeData): Processed data (with interpolations applied)
    """
    
    def __init__(
        self, 
        data: List[str], 
        des_dir: str,
        seed: int = 42, 
        boost_iter: int = 100,  
        thread_count: int = None

    ):
        super().__init__(datatype="tabular", data=data, des_dir=des_dir)
        self.seed = seed
        self.boost_iter = boost_iter
        self.thread_count = thread_count


    ###########
    # Helpers #
    ###########

    def load_tabular(self, tab_path: str) -> pd.DataFrame:
        """ Loads in a single tabular dataset. To ensure that no null values
            exists, local interpolation is conducted first to allow for more
            distribution-specific values.

        Returns:
            Tabular dataset (pd.DataFrame)
        """
        data = pd.read_csv(tab_path)

        # Retrieve corresponding schema path. Schemas as supposed to be
        # stored in the same directory as the dataset(s). At each root
        # classification, there can be more than 1 dataset, but all datasets
        # must have the same schema.
        schema_path = os.path.join(tab_path.parent, "schema.json")
        with open(schema_path, 'r') as s:
            schema = json.load(s)

        ##################################################
        # Edge Case: No seeding values for interpolation #
        ##################################################

        # [Cause]
        # This happens when there are feature columns within the original 
        # dataset that have no values at all (i.e. all NAN values). Being a 
        # NAN slice, CatBoost will not have any values to infer trends on, 
        # and will not be able to interpolate those aflicted features. 
        # CatBoost WILL NOT raise errors, since it deems NANs as valid 
        # types. As a result, even after inserting spacer columns (obtained 
        # from performing multiple feature alignment) on one-hot-encoded 
        # matrices, there will still be NAN values present.

        # [Problems]
        # Feeding NAN values into models & criterions for loss calculation 
        # will cause errors, breaking the FL training cycle and halting the 
        # grid.

        # [Solution]
        # Remove features corresponding to NAN slices entirely from dataset.
        # This allows a true representation of the participant's data to be
        # propagated down the pipeline, which can be caught & penalised
        # accordingly by the contribution calculator downstream.

        # Augment schema to cater to condensed dataset
        na_slices = data.columns[data.isna().all()].to_list()
        logging.debug(f"NA slices: {na_slices}")

        condensed_schema = {
            feature: d_type for feature, d_type in schema.items()
            if feature not in na_slices
        }
        condensed_data = data.dropna(axis='columns', how='all')
        assert set(condensed_schema.keys()) == set(condensed_data.columns)
        logging.debug(f"Condensed columns: {list(condensed_data.columns)}, length: {len(condensed_data.columns)}")

        preprocessor = Preprocessor(
            datatype=self.datatype,
            data=condensed_data, 
            schema=condensed_schema, 
            des_dir=self.des_dir,
            seed=self.seed,
            boost_iter=self.boost_iter,
            thread_count=self.thread_count
        )
        interpolated_data = preprocessor.interpolate()

        return interpolated_data


    def load_tabulars(self) -> pd.DataFrame:
        """ Loads in all tabular datasets found in the declared path sets

        Returns
            Aggregated tabular dataset (pd.DataFrame)
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            datasets = list(executor.map(self.load_tabular, self.data))

        # Assumption: At least one data path has been specified
        aggregated_df = pd.concat(
            datasets, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        logging.debug(f"Tag-unified Schema: {aggregated_df.dtypes.to_dict()}")

        return aggregated_df

    ##################
    # Core Functions #
    ##################

    def run(self) -> pd.DataFrame:
        """ Wrapper function that automates the tabular-specific preprocessing 
            of the declared datasets

        Returns
            Output (pd.DataFrame) 
        """
        if not self.is_processed():
            self.output = self.load_tabulars()

        return self.output