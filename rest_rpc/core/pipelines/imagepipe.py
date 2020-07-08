#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import logging
import os
from pathlib import Path
from string import Template
from typing import Dict, List

# Libs
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import minmax_scale, MinMaxScaler, LabelEncoder

# Custom
from rest_rpc.core.pipelines.basepipe import BasePipe
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################


########################################
# Data Preprocessing Class - ImagePipe #
########################################

class ImagePipe(BasePipe):
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
        des_dir (str): Destination directory to save data in
        data   (list(str)): Loaded data to be processed
        output (pd.DataFrame): Processed data (with interpolations applied)
    """

    def __init__(
        self, 
        data: List[str], 
        des_dir: str,
        use_grayscale: bool = True,
        use_alpha: bool = False,
        use_deepaugment: bool = True,
    ):
        super().__init__(data=data, des_dir=des_dir)
        self.use_grayscale = use_grayscale
        self.use_alpha = use_alpha
        self.use_deepaugment = use_deepaugment

    ###########
    # Helpers #
    ###########

    def load_image(
        self, 
        img_class: str, 
        img_path: str
    ) -> Dict[str, int]:
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

            img_format = Template("$color$alpha")
            color = "L" if self.use_grayscale else "RGB"
            alpha = "A" if self.use_alpha else ""
            img_format = img_format.substitute(color=color, alpha=alpha)
            formatted_img = img.convert(img_format)

            # Temporary implementation until full generalisation is achieved
            pix_vals = np.asarray(formatted_img).reshape((1, width * height))
            logging.debug(f"Pixel Values: {pix_vals}, {pix_vals.shape}")
            
        pix_map = pd.DataFrame(data=pix_vals, columns=pix_col_names)
        pix_map['target'] = img_class

        return pix_map


    def load_images(self) -> pd.DataFrame:
        """ Loads in all images found in the declared path sets

        Returns
            Output (pd.DataFrame)
        """
        for img_class, img_paths in self.data:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                class_images = list(executor.map(
                    lambda x: self.load_image(img_class, img_path=x), 
                    img_paths
                ))

        aggregated_df = pd.concat(class_images)
        
        # Ensure that all classes are numerically represented
        labelencoder = LabelEncoder()
        aggregated_df['target'] = labelencoder.fit_transform(
            aggregated_df['target'].astype('category')
        )

        return aggregated_df


    def apply_deepaugment(self):
        """ Apply AutoML methods to search for the appropriate preprocessing
            operations to use for transforming the images
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
        aggregated_df = self.load_images()
        self.apply_deepaugment()

        self.output = PipeData()
        self.output.update_data('image', aggregated_df)

        return self.output
