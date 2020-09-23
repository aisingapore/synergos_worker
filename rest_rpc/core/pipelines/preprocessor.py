#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import multiprocessing


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
from rest_rpc import app
from rest_rpc.core.pipelines.base import BasePipe
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################

BUFFER_FEATURE = "B" + "_"*5 + "#" # entirely arbitrarily chosen

cores_used = app.config['CORES_USED']

######################################################
# Data Preprocessing Class - Iterative Interpolation #
######################################################

class Preprocessor(BasePipe):
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
        schema (dict): Schema of the loaded dataframe. If not specified, schema
            will automatically be inferred from the declared dataset
        output (pd.DataFrame): Processed data (with interpolations applied)
    """

    def __init__(
        self,
        datatype: str, 
        data: pd.DataFrame,
        schema: Dict[str, str] = None,
        seed: int = 42, 
        boost_iter: int = 100, 
        des_dir: str = None, 
        thread_count: int = -1,
        allow_writing_files=False
    ):
        super().__init__(datatype=datatype, data=data, des_dir=des_dir)
        random.seed(seed)
        IPARAMS = {
            'iterations': boost_iter,
            'random_seed': seed,
            'thread_count': thread_count,
            'logging_level': 'Silent',
            'train_dir': des_dir,
            'allow_writing_files': allow_writing_files
        }
        self.__cat_interpolator = cat.CatBoostClassifier(**IPARAMS)
        self.__num_interpolator = cat.CatBoostRegressor(**IPARAMS)
        self.__seed = seed
        self.schema = (
            schema
            if schema
            else self.data.dtypes.apply(lambda x: x.name).to_dict()
        )
        logging.debug(f"Schema in Preprocessor: {self.schema}")

    ############        
    # Checkers #
    ############
    

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
        feature_vals = df[feature].copy()
        data = df.dropna(axis=1)
        data.loc[:, feature] = feature_vals

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
        null_data = df[df[feature].isnull()].dropna(axis=1)
        null_data[feature] = np.nan
        return null_data
    
    
    def isolate_uncertain_features(self):
        """ Identifies weak features for interpolation. Weak features are 
            defined by low significance (i.e. significance < 1).

        Returns:
            Ranked weak features (order of decreasing significance) (list)
        """
        # Evaluate significance of each column
        significances =  self.data.apply(self.calculate_significance)
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

        final_basis = basis.sort_index(axis=0)
            
        return final_basis[feature]
    

    def extract_image_metadata(self):
        """
        """
        if self.datatype == "image":

            # Hack: Use final feature column & parse height + width of each photo
            img_fmt, h_idx, w_idx = self.data.drop(columns='target').columns[-1].split('x')

            # Hack: Padded values are [0, ..., 0], where no. of '0's is len(img_fmt)
            pix_pad = tuple([0]*len(img_fmt))

            return pix_pad, int(h_idx), int(w_idx)


    def pad_images(self, df):
        """ Add in spacer values corresponding to the dimensions of each image.
            For example, for an 'RGBA' image has 10 pixel rows and 10 pixel 
            columns, whereby each pixel has 3 color channels + 1 alpha channel,
            dimension of the image will be (10, 10, 4). However, to facilitate
            alignment, .Originally need to cast to [Batch x Channels x Height x Width]
            # But for alignment's sake, flatten first, then recast after 
            # alignment is done

            "{img_format}x{h_idx}x{w_idx}"

        np.asarray(pd.DataFrame(np.asarray(img.convert('RGBA')).tolist()).values.tolist())
        """
        pix_pad, _, _ = self.extract_image_metadata()

        # df.fillna() does not work for non-scalars, hence, apply pads manually.
        for col in df.columns:
            if col != 'target':
                df[col] = df[col].apply(
                    lambda d: d if isinstance(d, tuple) else pix_pad
                )

        return df


    @staticmethod
    def pad_texts(df):
        """ Add in spacer values corresponding to each missing coordinate in the
            word vector.
        """
        return df.fillna(0)
    

    def align_dataset(self, dataset: np.ndarray, alignment_idxs: List[int]):
        """ Takes in a dataset & inserts null columns in accordance to MFA
            defined spacer indexes. Alignment indexes are sorted in ascending
            order.

        Args:
            dataset (th.Tensor): Data tensor to be augmented
            alignment_idxs (list(int)): Spacer indexes where header should
                insert null columns in order to properly align dataset to model
        Returns:
            Augmented dataset (th.Tensor)
        """
        logging.debug(f"Alignment indexes: {alignment_idxs}")

        aligned_dataset = dataset.copy()
        for idx in alignment_idxs:

            logging.debug(f"Current spacer index: {idx}")
            logging.debug(f"Before augmentation: size is {aligned_dataset.shape}")

            if self.datatype == "image":
                pix_pad, _, _ = self.extract_image_metadata()
                spacer_column = [pix_pad] * dataset.shape[0]
            else:
                spacer_column = [[0]] * dataset.shape[0]

            aligned_dataset = np.insert(
                aligned_dataset, 
                [idx], 
                spacer_column, 
                axis=1
            )

            logging.debug(f"After augmentation: size is {aligned_dataset.shape}")
        
        return aligned_dataset


    @staticmethod
    def align_header(header: List[str], alignment_idxs: List[int]):
        """ Takes in headers (i.e. list of features where order of arrangement
            matters) & inserts buffer features in accordance to MFA defined 
            spacer indexes. Alignment indexes are sorted in ascending
            order.

        Args:
            header (list(str)): Column/feature names to be aligned
            alignment_idxs (list(int)): Spacer indexes where header should
                insert null columns in order to properly align dataset to model
        Returns:
            Augmented dataset (th.Tensor)
        """
        augmented_headers = header.copy()
        for idx in alignment_idxs:
            augmented_headers.insert(idx, BUFFER_FEATURE + str(idx))

        return augmented_headers            
        

    def transform_defaults(
        self,
        action: str,
        scaler: Callable = minmax_scale,
        is_condensed: bool = True,
        X_alignments: List[int] = None,
        y_alignments: List[int] = None
    ):
        """ Applies default transformation procedures on tabular and text 
            datasets.

        Args:
            scaler   (func): Scaling function to be applied unto numerics
            is_condensed (bool): Whether to shrink targets to 2 classes
            X_alignments (list(int)): Alignment indexes for features from MFA
            y_alignments (list(int)): Alignment indexes for labels from MFA
        Return:
            X (np.ndarray)
            y (np.ndarray)
            X_header (list(str))
            y_header (list(str))
        """
        X, y, X_header, y_header = super().transform(
            action=action,
            scaler=scaler,   
            is_condensed=is_condensed
        )

        logging.debug(f"Transformed default X headers: {X_header} {len(X_header)}")
        logging.debug(f"Transformed default y headers: {y_header} {len(y_header)}")

        if X_alignments:
            X = self.align_dataset(X, X_alignments)
            X_header = self.align_header(X_header, X_alignments)

        if y_alignments:
            y = self.align_dataset(y, y_alignments)
            y_header = self.align_header(y_header, y_alignments)

        return X, y, X_header, y_header


    def transform_images(
        self, 
        action: str,
        is_condensed: bool = True,
        X_alignments: List[int] = None,
        y_alignments: List[int] = None
    ):
        """ Applies image-specific transformation procedures on a preprocessed
            image dataset

        Args:
            scaler   (func): Scaling function to be applied unto numerics
            is_condensed (bool): Whether to shrink targets to 2 classes
            X_alignments (list(int)): Alignment indexes for features from MFA
            y_alignments (list(int)): Alignment indexes for labels from MFA
        Return:
            X (np.ndarray)
            y (np.ndarray)
            X_header (list(str))
            y_header (list(str))
        """
        X, y, X_header, y_header = super().transform(
            action=action,
            scaler=None,    # deactivate default transformation
            is_sorted=False,# No sorting since it destroys convolution
            is_ohe=False,   # No OHE for image pixels
            is_condensed=is_condensed
        )
        formatted_X = np.array(X.tolist())

        _, h_idx, w_idx = self.extract_image_metadata()
        height = h_idx + 1
        width = w_idx + 1
        data_count = formatted_X.shape[0]

        if X_alignments:
            formatted_X = self.align_dataset(
                formatted_X, 
                X_alignments
            )
            X_header = self.align_header(X_header, X_alignments)

        formatted_X = formatted_X.reshape((data_count, -1, height, width))
        logging.debug(f"Formatted_X: {formatted_X}")

        if y_alignments:
            y = self.align_dataset(y, y_alignments)
            y_header = self.align_header(y_header, y_alignments)

        return formatted_X, y, X_header, y_header

    ##################    
    # Core Functions #
    ##################
    
    def interpolate(self, drop_duplicates=True):
        """ Performs feature-wise interpolations for loaded dataset across all
            weak features. Strength of a feature is defined by its significance.

            IMPORTANT!
            Due to the nature of the datasets, only tabular data will be allowed
            to be interpolated!

        Returns:
            Interpolated Data (pd.DataFrame)
        """
        if self.datatype == "tabular":
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

        else:
            raise RuntimeError(f"Interpolation not supported for datatype '{self.datatype}'!")


    def pad(self, drop_duplicates=True):
        """ Handles image & text-derived datasets by filling up unaligned
            segments with non-representation (eg. 0)

            IMPORTANT!
            Due to the nature of the datasets, only image & text datasets are
            allowed to be padded!

        Returns:
            Padded Data (pd.DataFrame)
        """
        if self.datatype == "image":
            logging.debug(f"Preprocessor's input data: {self.data}")
            self.output = self.pad_images(self.data)
        
        elif self.datatype == "text":
            self.output = self.pad_texts(self.data)

        else:
            raise RuntimeError(f"Padding not supported for datatype '{self.datatype}'!")

        if drop_duplicates:
            self.output = self.output.drop_duplicates().reset_index(drop=True)

        return self.output
        

    def run(self):
        """ Wrapper function that automates finetune preprocessing operations on
            the current dataset.

        Returns
            Output (pd.DataFrame) 
        """
        if self.datatype == "tabular":
            self.interpolate()

        elif self.datatype in ["image", "text"]:
            self.pad()

        else:
            raise RuntimeError(f"Datatype '{self.datatype}' not supported!")
        
        return self.output


    def transform(
        self, 
        action,
        scaler: Callable = minmax_scale,
        is_condensed: bool = True,
        X_alignments: List[int] = None,
        y_alignments: List[int] = None
    ):
        """ In addition to the standard transformation that pipelines can
            execute, this implements the auto-alignment procedure by augmenting 
            the X & y datasets using specified alignment indexes received from
            the TTP.

        Args:
            scaler   (func): Scaling function to be applied unto numerics
            is_condensed (bool): Whether to shrink targets to 2 classes
            X_alignments (list(int)): Alignment indexes for features from MFA
            y_alignments (list(int)): Alignment indexes for labels from MFA
        Returns:
            Aligned X_tensor (np.ndarray)
            Aligned X_tensor (np.ndarray)
            X_header         (list(str))
            y_header         (list(str))
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # When transforming dataset into an appropriate form based on the 
        # orchestration context, target labels are prematurely condensed before
        # their potential alignments.

        # [Problems]
        # Insertions of spacer columns AFTER condensing will result in y tensors
        # that have wrong label dimensions (i.e. [n, 2]), which raises the 
        # exception "RuntimeError: 1D target tensor expected, multi-target not 
        # supported". This error occurs because the criterion used from PyTorch 
        # during training MUST take in target values where Target: (N) where 
        # each value is 0≤targets[i]≤C−1.

        # [Solution]
        # Manually condense tensor AFTER inserting alignment spacers

        if self.datatype in ["tabular", "text"]:
            X, y, X_header, y_header = self.transform_defaults(
                action=action,
                scaler=scaler,
                is_condensed=False,  # DO NOT condense y labels first
                X_alignments=X_alignments,
                y_alignments=y_alignments
            )

        elif self.datatype == "image":
            X, y, X_header, y_header = self.transform_images(
                action=action,
                is_condensed=False,  # DO NOT condense y labels first
                X_alignments=X_alignments,
                y_alignments=y_alignments
            )

        else:
            raise ValueError(f"Specified Datatype {self.datatype} is not supported!")

        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # In order to manually condense tensor labels, the type of machine
        # learning operation has to first be known. For example, in a regression
        # problem, the targets are simply prediction values; these values cannot
        # be OHE-ed, compressed, or aligned. However, in a classification 
        # problem, there can be binary or multi-class classification. Even here
        # there is a distinct difference in treatment - binary class labels
        # exists as a single column of [0, 1] (i.e. compressed), while 
        # multi-class labels are to be OHE-ed (i.e. uncompressed with multiple
        # columns). Furthermore, in PyTorch criterions, classification labels
        # are expected to be casted into long values, while regression values
        # are expected to be casted into floats. 

        # [Problems]
        # Without explicitly stating the machine learning action to be executed,
        # labels are not handled properly, either being inappropriately 
        # expanded/compressed, or not expressed in the correct datatype/form. 
        # This results in criterion errors raised during the federated cycle 
        # over at the TTP

        # [Solution]
        # Add a parameter that explicitly specifies the machine learning 
        # operation to be handled, and take the appropriate action.

        X_tensor = th.Tensor(X)

        if action == 'regress':
            # No need for OHE or argmax
            y_tensor = th.Tensor(y).float() # not affected by OHE

        elif action == 'classify':
            if y.shape[1] < 2:
                raise RuntimeError('At least 2 classes MUST be specified!') 

            # Now condense labels into an appropriate form post-alignment
            y = np.argmax(y, axis=1) if is_condensed else y
            y_tensor = th.Tensor(y).long()
        
        else:
            raise ValueError(f"ML action {action} is not supported!")

        logging.debug(f"Casted y_tensor: {y_tensor} {y_tensor.type()}")

        return X_tensor, y_tensor, X_header, y_header

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