#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable

# Libs
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from tqdm import tqdm

# Custom
from rest_rpc.core.pipelines.abstract import AbstractPipe
from rest_rpc.core.pipelines.dataset import Singleton, PipeData

# Synergos logging
from SynergosLogger.init_logging import logging

##################
# Configurations #
##################

HEADERFILE = "clean_engineered_headers.txt"
DATAFILE = "clean_engineered_data.npy"
SCHEMAFILE = "clean_engineered_schema.json"

logging.info(f"base.py logged")

############################################
# Data Preprocessing Base Class - BasePipe #
############################################

class BasePipe(AbstractPipe):
    """ Contains baseline functionality to all pipelines. Other specific 
        pipelines will inherit all functionality for transforming processed data
        Extensions of this class overrides 1 key method `run` which are 
        responsible 

    Attributes:
        datatype (str): Type of process this pipeline is in charge of
        data (list): List of data units declared to be organised & aggregated
        des_dir (str): Destination directory to save data in
        output (pd.DataFrame): Processed data arranged into a dataframe
    """

    def __init__(self, datatype: str, data: list, des_dir: str):
        self.datatype = datatype
        self.data = data
        self.des_dir = des_dir
        self.output = self.load()
        #logging.notset(f"After initialisation, self.output: {self.output}")

    ############        
    # Checkers #
    ############

    def is_processed(self) -> bool:
        """ Checks if pipeline has ran at least once.

        Returns:
            True    if preprocessing operations have been performed
            False   otherwise
        """
        return self.output is not None


    def is_cached(self) -> bool:
        """ Checks if there are any cached operations that can be loaded. Used
            in tandem with `load()`.

        Returns:
            True    if cached operations have been detected
            False   otherwise
        """
        header_path = os.path.join(Path(self.des_dir).resolve(), HEADERFILE)
        des_path = os.path.join(Path(self.des_dir).resolve(), DATAFILE)
        schema_path = os.path.join(Path(self.des_dir).resolve(), SCHEMAFILE)
        return (
            os.path.isfile(header_path) and 
            os.path.isfile(des_path) and 
            os.path.isfile(schema_path)
        )

    ####################    
    # Helper Functions #
    ####################

    def parse_output(self) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
        """ Decomposes the final output into exportable elements. Essentially,
            all outputs must have headers (i.e. column/feature names), a data 
            commponent and a schema, al of which are targets for export.
        
        Returns
            headers (list(str))
            values (np.ndarray)
            schema (dict(str, str))
        """
        if not self.is_processed():
            logging.error("Data must first be processed!", Class=BasePipe.__name__, function=BasePipe.parse_output.__name__)
            raise RuntimeError("Data must first be processed!")
        
        headers = self.output.columns.to_list()
        values = self.output.to_numpy()
        schema = self.output.dtypes.apply(lambda x: x.name).to_dict()
        
        return headers, values, schema

    ##################
    # Core Functions #
    ##################

    def run(self):
        """ Runs preprocessing operations as defined in the pipeline. Please
            override this class in child classes for pipeline-specific 
            operations
        """
        raise NotImplementedError


    def transform(
        self,
        action: str,
        scaler: Callable = minmax_scale, 
        is_sorted: bool = True,
        is_ohe: bool = True,
        is_condensed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """ Converts interpolated data into a model-ready format 
            (i.e. One-hot encoding categorical variables) and splits final data
            into training and test data
            Note: For OHE features, despite being statistical convention to drop
                  the first OHE feature (i.e. feature class) of each categorical
                  variable (since it can be represened as a null matrix), in
                  this case, all OHE features will be maintained. This is
                  because in federated learning, as 1 worker's dataset possibly
                  contains only a small subset of the full feature space across
                  all workers, it is likely that:
                  1) Not all features are represented locally 
                     eg. [(X_0_0, X_1_1), (X_1_0, X_1_1), (X_2_0, X_2_1)] vs
                         [(X_0_0, X_1_1), (X_2_0, X_2_1)]
                  2) Not all feature classes are represented locally
                     eg. [(X_0_0, X_1_1), (X_1_0, X_1_1), (X_2_0, X_2_1)]
                         [(X_0_0       ), (X_1_0, X_1_1), (       X_2_1)]
                  Hence, there is a need to maintain full local feature coverage
                  multiple feature alignment will be conducted. However, MFA
                  will misalign if it does have all possible OHE features to
                  work with.
        Args:
            scaler     (func): Scaling function to be applied unto numerics
            condense   (bool): Whether to shrink targets to 2 classes
        Return:
            X           (np.ndarray)
            y           (np.ndarray)
            X_header    (list(str))
            y_header    (list(str))
        """
        if not self.is_processed():
            raise RuntimeError("Data must first be processed!")
        
        features = self.output.drop(columns=['target'])
        logging.debug(f"Feature datatypes: {features.dtypes.to_dict()}", Class=BasePipe.__name__, function=BasePipe.transform.__name__)

        if is_ohe:
            features = pd.get_dummies(features)

        if is_sorted:
            features = features.reindex(sorted(features.columns), axis=1)
            logging.debug(f"Sorted features: {features.columns}", Class=BasePipe.__name__, function=BasePipe.transform.__name__)
            logging.debug(f"Sorted datatypes: {features.dtypes.to_dict()}", Class=BasePipe.__name__, function=BasePipe.transform.__name__)

        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # Multiple feature alignment performed at TTP requires headers 
        # obtained from OHE-ed features as well as labels. However, based on
        # the machine learning operation to be executed, labels have to be 
        # handled differently.

        # [Problems]
        # Regression predictions are not categorical variables and should 
        # not be OHE-ed. Classification labels should be OHE-ed, and a 
        # unique set of headers should always be generated. This includes 
        # edge-cases like when there is only a single class.
        # eg. A single class of target 0 VS target 1 VS Regression target
        #
        #    target            target_0     <-- Unique header after OHE
        # 0       0   -->   0         1
        # 1       0   -->   1         1
        # 2       0   -->   2         1
        #
        #    target            target_1     <-- Unique header after OHE
        # 0       1   -->   0         1
        # 1       1   -->   1         1
        # 2       1   -->   2         1
        #
        #    target            target       <-- no change since No OHE
        # 0     1.0   -->   0     1.0
        # 1     2.0   -->   1     2.0
        # 2     3.0   -->   2     3.0

        # [Solution]
        # Explicitly cast regression targets as floats, while classification
        # targets as categories.

        if action == 'regress':
            targets = self.output[['target']].astype(dtype=float)
        elif action == 'classify':
            targets = self.output[['target']].astype(dtype='category')
        else:
            raise ValueError(f"ML action {action} is not supported!")
        
        logging.debug(f"Target datatypes: {targets.dtypes.to_dict()}", Class=BasePipe.__name__, function=BasePipe.transform.__name__)

        if not is_condensed:
            targets = pd.get_dummies(targets) # no effect on regression values

        feat_vals = scaler(features.values) if scaler else features.values
        target_vals = targets.values

        feat_header = features.columns.to_list()
        target_header = targets.columns.to_list()
        
        return (
            feat_vals, 
            target_vals, 
            feat_header, 
            target_header
        )
    

    def export(self) -> Tuple[str]:
        """ Exports the interpolated dataset.
            Note: Exported dataset is not one-hot encoded for extensibility
        
        Returns:
            Final filepath (str)
            Final schema path (str)
        """
        headers, values, schema = self.parse_output()

        os.makedirs(self.des_dir, exist_ok=True)

        header_path = os.path.join(Path(self.des_dir).resolve(), HEADERFILE)
        with open(header_path, 'w') as hp:
            json.dump(headers, hp)

        des_path = os.path.join(Path(self.des_dir).resolve(), DATAFILE)
        np.save(des_path, values)

        schema_path = os.path.join(Path(self.des_dir).resolve(), SCHEMAFILE)
        with open(schema_path, 'w') as sp:
            json.dump(schema, sp)

        return header_path, des_path, schema_path
        

    def offload(self):
        """ Converts pipeline outputs into a PipeData object, which lazyloads
            information to reduce memory footprint, as well as to abstract out
            tedious data combination management.

        Returns:
            Convert output (PipeData) 
        """
        if self.is_processed():
            header_path, des_path, schema_path = self.export()
            converted_output = PipeData()
            converted_output.update_data(
                datatype=self.datatype, 
                header_path=header_path, 
                data_path=des_path, 
                schema_path=schema_path
            )
            return converted_output

        else:
            logging.error("Pipeline needs to be ran first!", Class=BasePipe.__name__, function=BasePipe.offload.__name__)
            raise RuntimeError("Pipeline needs to be ran first!")


    def load(self) -> pd.DataFrame:
        """ Checks if there is a cached version of the output from a previous
            pipeline operation.

            IMPORTANT!
            This is based on the assumption that after the connection phase, no 
            further declaration or changes in data will be made/allowed in the
            system. With this assumption, this means that while there can be 
            many different tag combinations, each dataset corresponding to each
            tag only needs to be processed ONCE. Formatted data are exported. 
            Subsequent attempts to run the pipeline should load the cached data.
            This will reduce time spent on preprocessing.

        Returns:
            Loaded pre-formatted data (pd.DataFrame)
        """
        if self.is_cached():
            header_path = os.path.join(Path(self.des_dir).resolve(), HEADERFILE)
            des_path = os.path.join(Path(self.des_dir).resolve(), DATAFILE)
            schema_path = os.path.join(Path(self.des_dir).resolve(), SCHEMAFILE)

            cached_state = Singleton(header_path, des_path, schema_path)
            self.output = cached_state.data

            return self.output


    def reset(self) -> bool:
        """ Undo preprocessing effects

        Return:
            True    if operation is successful
            False   otherwise
        """
        self.output = None
        return self.output is None