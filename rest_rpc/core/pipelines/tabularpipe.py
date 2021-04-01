#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import json
import os
from pathlib import Path
from typing import List

# Libs
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom
from rest_rpc import app
from rest_rpc.core.pipelines.base import BasePipe
from rest_rpc.core.pipelines.preprocessor import Preprocessor
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

cores_used = app.config['CORES_USED']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("tabularpipe.py logged", Description="No Changes")

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
        thread_count: int = -1,
        allow_writing_files: bool = False
    ):
        super().__init__(datatype="tabular", data=data, des_dir=des_dir)
        self.seed = seed
        self.boost_iter = boost_iter
        self.thread_count = thread_count
        self.allow_writing_files = allow_writing_files

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

            logging.debug(
                f"No. of keys in schema: {schema} {len(schema)}", 
                ID_path=SOURCE_FILE,
                ID_class=TabularPipe.__name__, 
                ID_function=TabularPipe.load_tabular.__name__
            )

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
        logging.debug(
            f"NA slices: {na_slices}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabular.__name__
        )

        condensed_schema = {
            feature: d_type for feature, d_type in schema.items()
            if feature not in na_slices
        }
        condensed_data = data.dropna(axis='columns', how='all')
        
        logging.debug(
            f"Condensed schema: {condensed_schema}, length: {len(condensed_schema.keys())}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabular.__name__
        )
        logging.debug(
            f"Condensed columns: {list(condensed_data.columns)}, length: {len(condensed_data.columns)}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabular.__name__
        )
        logging.debug(
            f"Difference: {set(condensed_schema.keys()).symmetric_difference(set(condensed_data.columns))}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabular.__name__
        )
        assert set(condensed_schema.keys()) == set(condensed_data.columns)

        preprocessor = Preprocessor(
            datatype=self.datatype,
            data=condensed_data, 
            schema=condensed_schema, 
            des_dir=self.des_dir,
            seed=self.seed,
            boost_iter=self.boost_iter,
            thread_count=self.thread_count,
            allow_writing_files=self.allow_writing_files
        )
        interpolated_data = preprocessor.interpolate()

        logging.debug(
            f"After local interpolation: {interpolated_data.dtypes.to_dict()}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabular.__name__
        )

        return interpolated_data


    def load_tabulars(self) -> pd.DataFrame:
        """ Loads in all tabular datasets found in the declared path sets

        Returns
            Aggregated tabular dataset (pd.DataFrame)
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            datasets = list(executor.map(self.load_tabular, self.data))

        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # While concatenating categories, pandas will revert categorical
        # datatypes to integers if the no. of class labels are not the same.

        # [Problems]
        # This results in stray typing in the dataframe, which will lead to
        # problems downstream in schema-reliant operations.

        # [Solution]
        # Explicitly organise the schema and cast the dataframe accordingly.

        # Aggregate all schemas of all datasets
        aggregated_schema = {
            feature: datatype 
            for df in datasets 
            for feature, datatype in df.dtypes.to_dict().items()
        }

        # Assumption: At least one data path has been specified
        aggregated_df = pd.concat(
            datasets, 
            axis=0,
            sort=False
        ).drop_duplicates().reset_index(drop=True).astype(aggregated_schema)

        logging.debug(
            f"Tag-unified Schema: {aggregated_df.dtypes.to_dict()} {len(aggregated_df.dtypes.to_dict())}", 
            ID_path=SOURCE_FILE,
            ID_class=TabularPipe.__name__, 
            ID_function=TabularPipe.load_tabulars.__name__
        )

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