#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import os
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

# Libs
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Custom
from rest_rpc import app
from rest_rpc.core.pipelines.base import BasePipe
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################

logging = app.config['NODE_LOGGER'].synlog
logging.debug("imagepipe.py logged", Description="No Changes")

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
        super().__init__(datatype="image", data=data, des_dir=des_dir)
        self.use_grayscale = use_grayscale
        self.use_alpha = use_alpha
        self.use_deepaugment = use_deepaugment

    ###########
    # Helpers #
    ###########

    def parse_output(self) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
        """ In order for the pipelines to be symmetrical, images which are 
            higher dimensional objects are reshaped to (n, height * width, -1)
            ndarrays, and stored as a dataframe, where each column represents
            all values in a pixel. This function ensures that this adapted 
            structure can be properly transformed back into a numpy array when 
            necessary.

        Returns:
            headers (list(str))
            Image array (np.ndarray)
            schema (dict(str, str))
        """
        headers, values, schema = super().parse_output()
        formatted_values = np.array(values.tolist())
        return headers, formatted_values, schema


    def load_image(self, img_class: str, img_path: str) -> pd.DataFrame:
        """ Loads in a single image and retrieves its pixel values

        Args:
            img_class (str): Classification label of image
            img_path (str): Path to image
        Returns:
            Pixel Map (pd.DataFrame)
        """
        with Image.open(img_path) as img: 

            img_format = Template("$color$alpha")
            color = "L" if self.use_grayscale else "RGB"
            alpha = "A" if self.use_alpha else ""
            img_format = img_format.substitute(color=color, alpha=alpha)

            # Generate column names according to dimensions of image. This will
            # allow for auto-padding during feature alignment, both locally 
            # (between declared image datasets), and across the grid (between 
            # datasets amongst workers)
            width, height = img.size
            pix_col_names = [
                f"{img_format}x{h_idx}x{w_idx}"
                for h_idx in range(height)
                for w_idx in range(width)
            ]

            # Originally need to cast to [Batch x Channels x Height x Width]
            # But for the sake of alignment, flatten first. Allow Preprocessor
            # to handle the necessary operations for formatting the images for
            # use in WebsocketServerWorker
            # np.asarray(pd.concat([df1, df2]).values.tolist()).reshape(2, -1, 28, 28)
            formatted_img = img.convert(img_format)
            pix_vals = np.asarray(formatted_img).reshape(
                (1, height * width, -1)
            ).tolist()
            
        pix_map = pd.DataFrame(data=pix_vals, columns=pix_col_names)
        pix_map['target'] = img_class

        return pix_map


    def load_images(self) -> pd.DataFrame:
        """ Loads in all images found in the declared path sets

        Returns
            Output (pd.DataFrame)
        """
        all_images = []
        for img_class, img_paths in self.data:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                class_images = list(executor.map(
                    lambda x: self.load_image(img_class, img_path=x), 
                    img_paths
                ))

            all_images += class_images

        aggregated_df = pd.concat(all_images)
        aggregated_df['target'] = aggregated_df['target'].astype('category')

        return aggregated_df


    def apply_deepaugment(self, df):
        """ Apply AutoML methods to search for the appropriate preprocessing
            operations to use for transforming the images
        """
        return df

    ##################
    # Core Functions #
    ##################

    def run(self) -> pd.DataFrame:
        """ Wrapper function that automates the image-specific preprocessing of
            the declared datasets

        Returns
            Output (pd.DataFrame) 
        """
        if not self.is_processed():
            aggregated_df = self.load_images()
            self.output = self.apply_deepaugment(aggregated_df)

        return self.output
