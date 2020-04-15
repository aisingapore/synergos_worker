#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
import random
from pathlib import Path

# Libs
import numpy as np
import pandas as pd
import syft as sy
import torch as th
import tensorflow as tf
import xgboost as xgb
import catboost as cat
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from tqdm import tqdm

# Custom

##################
# Configurations #
##################


######################################################
# Data Preprocessing Class - Iterative Interpolation #
######################################################

class Preprocessor:
    """
    The Preprocessor class prepares a specified dataset for model fitting by
    applying a generalised process of interpolation that is non-data-specific,
    before finally applying certain encryption/noising methods to introduce a 
    sufficient level of noise such that the identity of an individual becomes
    statistically untraceable.

    Prerequisite: Data MUST have its labels headered as 'target'

    Args:
        data   (pd.DataFrame): Data to be processed
        schema         (dict): Datatypes of features found within dataset
        seed            (int): Seed to fix random state for testing consistency
        *args:
        **kwargs:
        
    Attributes:
        __cat_interpolator (CatBoost): CatBoost Classifier to impute categories
        __num_interpolator (CatBoost): CatBoost Regressor to impute numerics
        __seed                  (int): Seed to fix the state of CatBoost

        data   (pd.DataFrame): Loaded data to be processed
        schema         (dict): Schema of the loaded dataframe
        output (pd.DataFrame): Processed data (with interpolations applied)
    """

    def __init__(self, data, schema, 
                 seed=42, boost_iter=100, train_dir=None, thread_count=None):
        random.seed(seed)
        IPARAMS = {'iterations': boost_iter,
                   'random_seed': seed,
                   'thread_count': thread_count,
                   'logging_level': 'Silent',
                   'train_dir': train_dir}
        self.__cat_interpolator = cat.CatBoostClassifier(**IPARAMS)
        self.__num_interpolator = cat.CatBoostRegressor(**IPARAMS)
        self.__seed = seed
        self.data = data
        self.schema = schema
        self.output = None

    ############        
    # Checkers #
    ############
    
    def is_interpolated(self):
        """ Checks if interpolation has been performed

        Returns:
            True    if interpolation has been performed
            False   otherwise
        """
        return self.output is not None

    ####################    
    # Helper Functions #
    ####################

    @staticmethod
    def convert_to_safe_repr(df):
        """ Converts a specified dataframe into a safe & consistent
            representation for performing boosting interpolation. Returns a safe
            copy of the original dataframe
            Assumption: Each category for categorical features MUST be specified 
                        as numerics, or coercible strings.

        Args:
            df (pd.DataFrame): Data to be casted for boosting interpolation
        Returns:
            Safe representation df (pd.DataFrame)
        """
        safe_repr_df = df.copy()
        for feature in safe_repr_df.columns:

            safe_repr_df[feature] = pd.to_numeric(
                safe_repr_df[feature],
                errors='coerce'
            ).astype(
                {feature:'float64'}
            )
        
        return safe_repr_df
    

    @staticmethod
    def revert_repr(df, schema):
        """ Reverts a specified dataframe back into its original datatypes in
            accordance to a specified schema. Returns a reverted copy of the
            original dataframe
            Assumption: Each category for categorical features MUST be specified 
                        as numerics, or coercible strings.

        Args:
            df (pd.DataFrame): Data to be reverted after boosting interpolation
        Returns:
            Original representation df (pd.DataFrame)
        """
        reverted_df = df.copy()

        # Cast each feature into the correct data type
        for feature in reverted_df.columns:
            is_int = 'int' in schema[feature]
            is_cat = 'category' in schema[feature]

            # Integers & categorical variables are casted to integers
            if is_int or is_cat:
                reverted_df[feature] = reverted_df[feature].astype(
                    {feature:'float64'}
                ).astype(
                    {feature:'int64'}
                )

            # Floats are casted into Floats
            else:
                reverted_df[feature] = reverted_df[feature].astype(
                    {feature:'float64'}
                )
        
        return reverted_df


    @staticmethod
    def calculate_significance(values):
        """ Calculates the significance of a set of feature values. Significance
            is defined as the extent of nullity of the current value set, and is
            used as a metric to rank features to determine the order of
            interpolation.
            
        Args:
            values (pd.Series): 
                Values of a specified feature
        Returns:
            Significance of feature values (float)
        """
        null_count = values.isnull().sum()
        obs_count = len(values)
        significance = 1 - (null_count/obs_count)
        return significance
    
    
    @staticmethod
    def extract_complete_segments(df, feature):
        """ Extracts existing rows within dataset whereby the specified feature
            is defined. This will serve as the basis for interpolation of rows
            with missing values for said feature

        Args:
            df (pd.DataFame): Reference dataframe to extract defined segment
            feature    (str): Feature relevant to extracted segment
        Returns:
            Feature-complete segment (pd.DataFrame)
        """
        # Replace all "?" & "-9.0" with NAN values
        # (This operation is unique to dataset, subjected to changes)
        data = df.replace(
            [-9.0, '?'], np.nan
        ).replace(
            {'ca': 9, 'slope': 0, 'thal': [1,2,5]}, np.nan
        )
        feature_vals = data[feature]
        data = data.dropna(axis=1)
        data[feature] = feature_vals
        feature_complete_segment = data[~data[feature].isnull()]
        return feature_complete_segment.dropna(axis=1)
    
        
    @staticmethod
    def extract_null_rows(df, feature):
        """ Extracts rows that require interpolation for the current feature.

        Args:
            df (pd.DataFame): Reference dataframe to extract null rows
            feature    (str): Feature relevant to extracted segment
        Returns:
            Null data (w.r.t specified feature) (pd.DataFrame)
        """
        # Replace all "?" & "-9.0" with NAN values
        # (This operation is unique to dataset, subjected to changes)
        data = df.replace(
            [-9.0, '?'], np.nan
        ).replace(
            {'ca': 9, 'slope': 0, 'thal': [1,2,5]}, np.nan
        )
        null_data = data[data[feature].isnull()].dropna(axis=1)
        null_data[feature] = np.nan
        return null_data
    
    
    def isolate_uncertain_features(self):
        """ Identifies weak features for interpolation. Weak features are 
            defined by low significance (i.e. significance < 1).

        Returns:
            Ranked weak features (order of decreasing significance) (list)
        """
        # Replace all "?" & "-9.0" with NAN values
        # (This operation is unique to dataset, subjected to changes)
        data = self.data.replace(
            [-9.0, '?'], np.nan
        ).replace(
            {'ca': 9, 'slope': 0, 'thal': [1,2,5]}, np.nan
        )
        # Evaluate significance of each column
        significances = data.apply(self.calculate_significance)
        # Filter out features with low significance
        low_sig_features = significances[significances < 1]
        # Sort according to significance, in descending order
        low_sig_features = low_sig_features.sort_values(ascending=False)
        # Extract corresponding columns names
        sorted_low_sig_cols = low_sig_features.index.tolist()
        return sorted_low_sig_cols
    
    
    def interpolate_feature(self, df, feature):            
        """ Interpolates missing values of the specified feature within the
            specified data. Interpolation is performed by training a CatBoost
            Model on defined segements of the data w.r.t to said feature, and
            predicting on undefined segments.

        Args:
            df (pd.DataFame): Reference dataframe to interpolation
            feature    (str): Feature relevant to interpolation
        Returns:
            Interpolated values of specified feature (pd.Series)
        """
        # Extract complete rows to be used as basis for interpolation
        basis = self.extract_complete_segments(df, feature)
        
        # Extract interpolatable data & ensure symmetry to basis
        null_data = self.extract_null_rows(df, feature)
        null_data = null_data[basis.columns]

        null_idxs = null_data.index.tolist()
        
        spacings = len(max(self.schema.keys()))
        pbar = tqdm(total=len(null_idxs), 
                    desc='Interpolating Feature: {:^{}}'.format(feature, spacings), 
                    leave=True)
        while len(null_idxs) > 0:
            # Initialise the appropriate interpolator
            interpolator = self.__num_interpolator
            if self.schema[feature] == 'category':
                interpolator = self.__cat_interpolator

            selected_null_idx = random.choice(null_idxs)
            curr_null_row = null_data.loc[selected_null_idx].copy()
            
            try:
                basis_X = basis.drop(columns=[feature])
                basis_y = basis[feature]
                interpolator.fit(basis_X.values, basis_y.values)
                null_X = curr_null_row.drop(labels=[feature])
                interpolated_value = interpolator.predict(null_X.values)[0]
            except Exception:
                interpolated_value = basis[feature].median()

            # Update the current row with the interpolated result
            curr_null_row[feature] = interpolated_value
            basis.loc[selected_null_idx] = curr_null_row

            null_idxs.remove(selected_null_idx)
            pbar.update(1)
        pbar.close()
        """
        # Cast basis into the correct data type
        feature_dtype = {feature: self.schema[feature]}
        is_int = 'int' in self.schema[feature]
        is_cat = 'category' in self.schema[feature]
        # Ensure that all values are properly represented first
        if is_int or is_cat:
            basis = basis.astype({feature:'float64'}).astype({feature:'int64'})
        else:
            basis = basis.astype({feature:'float64'})
        final_basis = basis.astype(feature_dtype).sort_index(axis=0)
        """
        final_basis = basis.sort_index(axis=0)
            
        return final_basis[feature]
    
    ##################    
    # Core Functions #
    ##################
    
    def interpolate(self, drop_duplicates=True, encrypt=False):
        """ Performs feature-wise interpolations for loaded dataset across all
            weak features. Strength of a feature is defined by its significance.

        Returns:
            Interpolated Data (pd.DataFrame)
        """
        self.output = self.convert_to_safe_repr(self.data)
        
        uncertain_features = self.isolate_uncertain_features()
        
        for u_feat in uncertain_features:
            interpolated_values = self.interpolate_feature(self.output, u_feat)
            self.output[u_feat] = interpolated_values
            
        self.output = self.revert_repr(self.output, self.schema)
        self.output = self.output.astype(self.schema)
        
        if drop_duplicates:
            self.output = self.output.drop_duplicates().reset_index(drop=True)

        return self.output
    
    
    def transform(self, scaler=minmax_scale, condense=True):
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
        if not self.is_interpolated():
            raise RuntimeError("Data must first be interpolated!")
        
        features = self.output.drop(columns=['target'])
        ohe_features = pd.get_dummies(features)#, drop_first=True)
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
    
    
    def export(self, des_dir='.'):
        """ Exports the interpolated dataset.
            Note: Exported dataset is not one-hot encoded for extensibility
        
        Args:
            des_dir (str): Destination directory to save data in
        Returns:
            Final filepath (str)
        """
        filename = "clean_engineered_data.csv"
        des_path = os.path.join(Path(des_dir).resolve(), filename)

        try:
            os.makedirs(des_path)
        except FileExistsError:
            # directory already exists
            pass

        clean_engineered_data = self.output
        clean_engineered_data.to_csv(path_or_buf=des_path,
                                     sep=',', 
                                     index=False,
                                     encoding='utf-8',
                                     header=True)
        return des_path
        
    
    def reset(self):
        """ Resets interpolations preprocessing

        Return:
            True    if operation is successful
            False   otherwise
        """
        self.output = None
        return self.output is None
        
#########        
# Tests #
#########

if __name__ == "__main__":
    # Read in a dataset
    DATA_HEADERS = ['age', 'sex', 
                    'cp', 'trestbps', 
                    'chol', 'fbs', 
                    'restecg', 'thalach', 
                    'exang', 'oldpeak', 
                    'slope', 'ca', 
                    'thal', 'target']
    schema = {'age': 'int32',
              'sex': 'category', 
              'cp': 'category', 
              'trestbps': 'int32', 
              'chol': 'int32', 
              'fbs': 'category', 
              'restecg': 'category', 
              'thalach': 'int32', 
              'exang': 'category', 
              'oldpeak': 'float64', 
              'slope': 'category', 
              'ca': 'category', 
              'thal': 'category', 
              'target': 'category'}
    DATA_HUNGARIAN_PATH = Path("./data/raw/reprocessed.hungarian.data").resolve()
    hungarian_data = pd.read_csv(DATA_HUNGARIAN_PATH, 
                                 sep=' ', 
                                 names=DATA_HEADERS)
    print(hungarian_data)

    # Testing preprocessor functionality
    preprocessor = Preprocessor(hungarian_data, schema)
    preprocessor.interpolate()
    print(preprocessor.output)
    print(preprocessor.transform())
    #print(preprocessor.export('./data'))