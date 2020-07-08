#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import abc
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

# Libs
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from tqdm import tqdm

# Custom
from rest_rpc.core.pipelines.abstractpipe import AbstractPipe
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################


####################################################
# Data Preprocessing Abstract Class - AbstractPipe #
####################################################

class BasePipe(AbstractPipe):
    """ Contains baseline functionality to all pipelines. Other specific 
        pipelines will inherit all functionality for transforming processed data
        Extensions of this class overrides 1 key method `run` which are 
        responsible 

    Attributes:
        des_dir (str): Destination directory to save data in
        data (list): List of data units declared to be organised & aggregated
        output (pd.DataFrame): Processed data arranged into a dataframe
    """

    def __init__(self, data: list, des_dir: str):
        self.des_dir = des_dir
        self.data = data
        self.output = None

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

    ####################    
    # Helper Functions #
    ####################


    ##################
    # Core Functions #
    ##################

    def run(self):
        """ Runs preprocessing operations as defined in the pipeline. Please
            override this class in child classes for pipeline-specific 
            operations
        """
        raise NotImplementedError


    def transform(self, 
        scaler=minmax_scale, 
        condense=True
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
            condense   (bool): Whether to shrink targets to 2 classes (not 5)
        Return:
            X           (np.ndarray)
            y           (np.ndarray)
            X_header    (list(str))
            y_header    (list(str))
        """
        if not self.is_processed():
            raise RuntimeError("Data must first be processed!")
        
        features = self.output.drop(columns=['target'])
        ohe_features = pd.get_dummies(features)
        ohe_feat_vals = scaler(ohe_features.values)
        
        targets = self.output[['target']].copy()
        ohe_targets = pd.get_dummies(targets)
        if condense:
            targets.loc[:,'target'] = targets.target.apply(lambda x: int(x > 0))
            ohe_targets = pd.get_dummies(targets, drop_first=True)
        ohe_target_vals = ohe_targets.values

        ohe_feat_header = ohe_features.columns.to_list()
        ohe_target_header = ohe_targets.columns.to_list()
        
        return (
            ohe_feat_vals, 
            ohe_target_vals, 
            ohe_feat_header, 
            ohe_target_header
        )
    
    
    def export(self) -> str:
        """ Exports the interpolated dataset.
            Note: Exported dataset is not one-hot encoded for extensibility
        

        Returns:
            Final filepath (str)
        """
        filename = "clean_engineered_data.csv"
        des_path = os.path.join(Path(self.des_dir).resolve(), filename)
        os.makedirs(des_path.parent, exist_ok=True)

        clean_engineered_data = self.output
        clean_engineered_data.to_csv(path_or_buf=des_path,
                                     sep=',', 
                                     index=False,
                                     encoding='utf-8',
                                     header=True)
        return des_path
        
    
    def reset(self) -> bool:
        """ Resets interpolations preprocessing

        Return:
            True    if operation is successful
            False   otherwise
        """
        self.output = None
        return self.output is None