#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List

# Libs
import pandas as pd

# Custom


##################
# Configurations #
##################


#################################
# Data Abstraction Class - Data #
#################################

class PipeData:
    """ Abstraction class for organising datasets of different datatypes during
        preprocessing. This class facilitates the implementation of localised
        sources, where each source is self-describing.

        * Lazy loading coming soon!

    Attributes:
        # Private Attributes
        __SUPPORTED_DATATYPES (list(str)): List of all supported datatypes
        
        # Public Attributes
        data (dict): Stores all data segments for on the fly aggregation
    """

    def __init__(self):
        self.__SUPPORTED_DATATYPES = ['tabular', 'image', 'text']
        self.data = {}
        self.schemas = {}


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
    def info(self):
        return {
            _type: self.__combine_data(df_list) 
            for _type, df_list in self.data.items()
        }

    @property
    def tabulars(self):
        return self.data.get('tabular', [])

    @property
    def images(self):
        return self.data.get('image', [])

    @property
    def texts(self):
        return self.data.get('text', [])

    ###########    
    # Setters #
    ###########

    def update_data(self, datatype, data):
        assert datatype in self.__SUPPORTED_DATATYPES
        curr_archive = self.data.get(datatype, [])
        curr_archive.append(data)
        self.data[datatype] = curr_archive
        
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

    @staticmethod
    def __combine_data(df_list):
        return pd.concat(
            df_list, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)