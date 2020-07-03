#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Dict, List

# Libs
import pandas as pd
from PIL import Image
from sklearn.preprocessing import minmax_scale, MinMaxScaler

# Custom

##################
# Configurations #
##################


########################################
# Data Preprocessing Class - ImagePipe #
########################################

class ImagePipe:
    """
    The ImagePipe class implement preprocessing tasks generalised for handling
    image data. The general workflow is as follows:
    1) Converts images into .csv format, with each pixel arranged in a single
       row, alongside annotated target labels
    2) Downscales images to lowest common denominator of all parties in the
       federated grid.
    3) Augment each image to bring out best local features (Automl: DeepAugment)
    4) Convert numpy to torch tensors for dataloading

    Prerequisite: Data MUST have its labels headered as 'target'

    Attributes:
        __seed (int): Seed to fix the random state of processes

        data   (dict(str)): Loaded data to be processed
        output (pd.DataFrame): Processed data (with interpolations applied)
    """
    def __init__(self, data: Dict[str,str], seed=42):
        self.data = data
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

    ###########
    # Helpers #
    ###########

    def load_image(self, img_class: str, img_path: str) -> Dict[str, int]:
        """ Loads in a single image and retrieves its pixel values

        Args:
            img_class (str): Classification label of image
            img_path (str): Path to image
        Returns:
            Pixel Map (dict(str, int))
        """
        with Image.open(img_path) as img: 

            # Generate column names according to dimensions of image. This will
            # allow for auto-padding during feature alignment, both locally 
            # (between declared image datasets), and across the grid (between 
            # datasets amongst workers)
            width, height = img.size
            pix_col_names = [
                f"{h_idx}x{w_idx}"
                for h_idx in range(height)
                for w_idx in range(width)
            ]

            grayscaled_img = img.convert('LA')       # Single color channel
            pix_val = list(grayscaled_img.getdata()) # (color, alpha)
            pix_val_flat = [sets[0] for sets in pix_val]

        pix_map = dict(zip(pix_col_names, pix_val_flat))
        pix_map.update({'target': img_class})

        return pix_map


    def load_images(self) -> pd.DataFrame:
        """ Loads in all images found in the declared path sets

        Returns
            Output (pd.DataFrame)
        """
        singleton_images = []
        for img_class, img_paths in self.data.items():

            with concurrent.futures.ThreadPoolExecutor() as executor:
                class_images = list(executor.map(
                    lambda x: self.load_image(img_class, img_path=x), 
                    img_paths
                ))
            
            singleton_images.extend(class_images)

        self.output = pd.DataFrame.from_records(singleton_images)

        return self.output


    def apply_deepaugment(self):
        """
        """
        pass

    ##################
    # Core Functions #
    ##################

    def run(self) -> pd.DataFrame:
        """ Wrapper function that automates the image-specific preprocessing of
            the declared datasets

        Returns
            Output (pd.DataFrame) 
        """
        self.load_images()
        self.apply_deepaugment()
        return self.output


    def transform(self, scaler=minmax_scale, condense=False):
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
            raise RuntimeError("Data must first be loaded first!")
        
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
        Path(des_path).mkdir(parents=True, exist_ok=True)

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
