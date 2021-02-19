#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import json
import logging
from typing import Dict, List, Union

# Libs
import numpy as np
import pandas as pd

# # Synergos logging
# from SynergosLogger.init_logging import logging
# Custom
from config import NODE_LOGGER

##################
# Configurations #
##################

logging.info(f"dataset.py logged")

######################################
# Data Abstraction Class - Singleton #
######################################

class Singleton:
    """ 
    Helper class for implementing lazy loading

    Attributes:
        features (str): Path to file containing all feature names
        source (str): Path to file containing source values
        types (str): Path to file containing all datatypes
    """

    def __init__(self, features, source, types):
        self.features = features
        self.source = source
        self.types = types

    @property
    def header(self) -> List[str]:
        """ Loads in exported schema of the current dataset

        Returns:
            Headers (list(str))
        """
        with open(self.features, 'r') as fn:
            header = json.load(fn)
        return header
        
    @property
    def schema(self) -> Dict[str, str]:
        """ Loads in exported schema of the current dataset

        Returns:
            Schema (dict(str, str))
        """
        with open(self.types, 'r') as dt:
            schema = json.load(dt)
        return schema

    @property
    def values(self) -> np.ndarray:
        """ Loads in exported schema of the current dataset

        Returns:
            Values (np.ndarray))
        """
        values = np.load(self.source, allow_pickle=True)
        return values

    @property
    def data(self) -> pd.DataFrame:
        """ Loads in values stored at the specified source as a dataframe &
            casts the feature headers into stipulated datatypes as specified
            under the types path

        Returns:
            Dataset (pd.DataFrame)
        """
        dataset = pd.DataFrame(data=self.values, columns=self.header)
        return dataset.astype(dtype=self.schema)

#############################################
# Data Abstraction Class - ComplexSingleton #
#############################################

class ComplexSingleton(Singleton):
    """ 
    Helper class for implementing lazy loading on data structures with higher
    dimensionality. Mainly used for encapsulating image data.

    Attributes:
        features (str): Path to file containing all feature names
        source (str): Path to file containing source values
        types (str): Path to file containing all datatypes
    """

    def __init__(self, features, source, types):
        super().__init__(features, source, types)

    @property
    def data(self) -> pd.DataFrame:
        """ Loads in values stored at the specified source as a dataframe &
            casts the feature headers into stipulated datatypes as specified
            under the types path. However, to ensure hashability of internal
            dimensions, cast all elements within the dataframe into tuples.

        Returns:
            Hashable Dataset (pd.DataFrame)
        """
        dataset = super().data
        for col in dataset.columns:
            try:
                if col != 'target':
                    dataset[col] = dataset[col].apply(tuple)
            except TypeError:
                logging.warning("Non-tuple when tried to be casted into tuple throws error", Class=ComplexSingleton.__name__, function=ComplexSingleton.data.__name__)
                # Skip any non-iterable elements
                pass

        return dataset

#####################################
# Data Abstraction Class - PipeData #
#####################################

class PipeData:
    """ 
    Abstraction class for organising datasets of different datatypes during
    preprocessing. This class facilitates the implementation of localised
    sources, where each source is self-describing.

    [Update]
    Main goal of this class is to allow quantized preprocessing, whereby every 
    declared tagged dataset can be preprocessed with their own custom set of 
    operations. Right now, quantized operations are supported on the assumption
    that all declared datasets are of the same type. In future, this assumption
    will be dismissed.

    Attributes:
        # Private Attributes
        __SUPPORTED_DATATYPES (list(str)): List of all supported datatypes
        
        # Public Attributes
        data (dict): Stores all data segments for on the fly aggregation
    """

    def __init__(self):
        self.__SUPPORTED_DATATYPES = ['tabular', 'image', 'text']
        self.data = {}


    def __add__(self, other):
        """ Automates the consolidation of differently typed datasets aggregated
            across different pipelines

        Args:
            other (PipeData)
        Returns:
            New aggregated data (PipeData)
        """
        new_pipedata = PipeData()
        
        for datatype in self.__SUPPORTED_DATATYPES:
            data_1 = self.data.get(datatype, []) 
            data_2 = other.data.get(datatype, [])
            new_pipedata.data[datatype] = data_1 + data_2

        return new_pipedata


    def __radd__(self, other):
        """ Essential for compatibility with bulk operations (i.e. sum)

        Args:
            other (PipeData)
        Returns:
            New aggregated data (PipeData)
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    ###########
    # Getters #
    ###########

    @property
    def tabulars(self):
        """ Retrieves all lazy loaded singletons corresponding to tabular data

        Returns:
            Tabular Singletons (list(Singleton)) 
        """
        return self.data.get('tabular', [])

    @property
    def images(self):
        """  Retrieves all lazy loaded singletons corresponding to image data

        Returns:
            Image Singletons (list(Singleton)) 
        """
        return self.data.get('image', [])

    @property
    def texts(self):
        """  Retrieves all lazy loaded singletons corresponding to text data

        Returns:
            Text Singletons (list(Singleton)) 
        """
        return self.data.get('text', [])

    ###########    
    # Setters #
    ###########

    def update_data(
        self, 
        datatype: str, 
        header_path: str, 
        data_path: str, 
        schema_path: str
    ) -> Union[Singleton, ComplexSingleton]:
        """ Takes in export paths to attributes that compose a dataset, and
            encapsulates them as singleton data structures for lazy loading

        Args:
            datatype: str, 
            header_path: str, 
            data_path: str, 
            schema_path: str
        Returns:
            Encapsulated data (Singleton or ComplexSingleton)
        """
        assert datatype in self.__SUPPORTED_DATATYPES
        curr_archive = self.data.get(datatype, [])

        if datatype == "image":
            dataset = ComplexSingleton(header_path, data_path, schema_path)
        else:
            dataset = Singleton(header_path, data_path, schema_path)
        
        curr_archive.append(dataset)
        self.data[datatype] = curr_archive

        return dataset
        
    ############    
    # Checkers #
    ############

    def has_tabulars(self):
        return bool(self.data.get('tabular', []))

    def has_images(self):
        return bool(self.data.get('images', []))

    def has_texts(self):
        return bool(self.data.get('text', []))

    ###########
    # Helpers #
    ###########

    def __load_data(self, singletons: List[Singleton]) -> List[pd.DataFrame]:
        """ Loads in all singleton datasets from source efficiently.

        Args:
            singletons (list(Singleton)): List of singleton objects to be loaded
        Returns:
            List of datasets (list(pd.DataFrame))
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            datasets = list(executor.map(lambda x: x.data, singletons))
        return datasets


    def __combine_data(self, singletons: List[Singleton]) -> pd.DataFrame:
        """ Loads in all singleton datasets and aggregates them into a single
            dataframe.

        Args:
            singletons (list(Singleton)): List of singleton objects to be loaded
        Returns:
            Combined dataset (pd.DataFrame)
        """
        df_list = self.__load_data(singletons)
        logging.debug(f"DF List: {df_list}", Class=PipeData.__name__, function=PipeData.__combine_data.__name__)

        return pd.concat(
            df_list, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

    ##################
    # Core Functions #
    ##################

    def compute(self) -> Dict[str, pd.DataFrame]:
        """ Resolves and load datasets according to filepaths and concatenate
            them into a single dataframe representative of each supported type
            of data (i.e. tabular, image, text)

        Return:
            Combined meta datasets (dict(str, pd.DataFrame))
        """
        return {
            _type: self.__combine_data(singletons) 
            for _type, singletons in self.data.items()
            if singletons 
        }