#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import abc
from typing import Callable

# Libs


# Custom

##################
# Configurations #
##################


####################################################
# Data Preprocessing Abstract Class - AbstractPipe #
####################################################

class AbstractPipe(abc.ABC):

    @abc.abstractmethod
    def run(self):
        """ Runs preprocessing operations as defined in the pipeline
        
            As AbstractPipe implies, you should never instantiate this class by
            itself. Instead, you should extend AbstractPipe in a new class which
            instantiates `run`.
        """
        pass

    
    @abc.abstractmethod
    def transform(self, scaler: Callable, condense: bool):
        """ Converts interpolated data into a model-ready format 
            (i.e. One-hot encoding categorical variables) and splits final data
            into training and test data

            As AbstractPipe implies, you should never instantiate this class by
            itself. Instead, you should extend AbstractPipe in a new class which
            instantiates `transform`.

        Args:
            scaler     (func): Scaling function to be applied unto numerics
            condense   (bool): Whether to shrink targets to 2 classes (not 5)
        """
        pass
    
    
    @abc.abstractmethod
    def export(self, des_dir: str):
        """ Exports the interpolated dataset.
            Note: Exported dataset is not one-hot encoded for extensibility
        
            As AbstractPipe implies, you should never instantiate this class by
            itself. Instead, you should extend AbstractPipe in a new class which
            instantiates `export`.

        Args:
            des_dir (str): Destination directory to save data in
        """
        pass
        
    
    @abc.abstractmethod
    def reset(self):
        """ Resets interpolations preprocessing

            As AbstractPipe implies, you should never instantiate this class by
            itself. Instead, you should extend AbstractPipe in a new class which
            instantiates `reset`.
        """
        pass
