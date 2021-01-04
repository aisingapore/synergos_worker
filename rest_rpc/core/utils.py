#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Union

# Libs
import jsonschema
import numpy as np
import pandas as pd
import torch as th
from flask import jsonify, request
from flask_restx import fields
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score, 
    roc_curve,
    roc_auc_score, 
    auc, 
    precision_recall_curve, 
    precision_score,
    recall_score,
    f1_score, 
    confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelBinarizer
from tinydb import TinyDB, Query, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tinyrecord import transaction
from tinydb_serialization import SerializationMiddleware
from tinydb_smartcache import SmartCacheTable

# Custom
from rest_rpc import app
from rest_rpc.core.datetime_serialization import (
    DateTimeSerializer, 
    TimeDeltaSerializer
)

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
payload_template = app.config['PAYLOAD_TEMPLATE']

####################
# Helper Functions #
####################

def construct_combination_key(expt_id, run_id):
    return str((expt_id, run_id))

############################################
# REST Response Formatting Class - Payload #
############################################

class Payload:
    """ Helper class to standardise response formatting for the REST-RPC service
        in order to ensure compatibility between the TTP's & Workers' Flask
        interfaces

    Attributes:
        # Private Attributes
        __template (dict): Configured payload template
        # Public Attributes
        subject (str): Topic of data in payload (i.e. name of table accessed)
    
    Args:
        subject (str): Topic of data in payload (i.e. name of table accessed)
        namespace (flask_restx.Namespace): Namespace API to construct models in
        model (flask_restx.Model): Seeding model to propagate
    """
    def __init__(self, subject, namespace, model):
        self.__template = payload_template.copy()
        self.subject = subject

        payload_model = namespace.model(
            name="payload",
            model={
                'apiVersion': fields.String(required=True),
                'success': fields.Integer(required=True),
                'status': fields.Integer(required=True),
                'method': fields.String(),
                'params': fields.Nested(
                    namespace.model(
                        name="route_parameters",
                        model={
                            'project_id': fields.String(),
                            'expt_id': fields.String(),
                            'run_id': fields.String(),
                        }
                    ),
                    skip_none=True
                )
            }
        )
        self.singular_model = namespace.inherit(
            "payload_single",
            payload_model,
            {'data': fields.Nested(model, required=True, skip_none=True)}
        )
        self.plural_model = namespace.inherit(
            "payload_plural",
            payload_model,
            {
                'data': fields.List(
                    fields.Nested(model, skip_none=True), 
                    required=True
                )
            }
        )

    def construct_success_payload(self, status, method, params, data):
        """ Automates the construction & formatting of a payload for a
            successful endpoint operation 
        Args:
            status (int): Status code of method of operation
            method (str): Endpoint operation invoked
            params (dict): Identifiers required to start endpoint operation
            data (list or dict): Data to be moulded into a response
        Returns:
            Formatted payload (dict)
        """
        
        def format_document(document, kind):

            def encode_datetime_objects(document):
                datetime_serialiser = DateTimeSerializer()
                document['created_at'] = datetime_serialiser.encode(document['created_at'])
                return document

            def annotate_document(document, kind):
                document['doc_id'] = document.doc_id
                document['kind'] = kind
                return document

            encoded_document = encode_datetime_objects(document)
            annotated_document = annotate_document(encoded_document, kind)
            return annotated_document

        self.__template['success'] = 1
        self.__template['status'] = status
        self.__template['method'] = method
        self.__template['params'] = params
        
        if isinstance(data, list):
            formatted_data = []
            for record in data:
                formatted_record = format_document(record, kind=self.subject)
                formatted_data.append(formatted_record)
        else:
            formatted_data = format_document(data, kind=self.subject)
                
        self.__template['data'] = formatted_data

        jsonschema.validate(self.__template, schemas['payload_schema'])

        return self.__template      

#####################################
# Base Data Storage Class - Records #
#####################################

class Records:
    """ 
    Automates CRUD operations on a structured TinyDB database. Operations are
    atomicised using TinyRecord transactions, queries are smart cahced

    Attributes:
        db_path (str): Path to json source
    
    Args:
        db_path (str): Path to json source
        *subjects: All subject types pertaining to records
    """
    def __init__(self, db_path=db_path):
        self.db_path = db_path

    ###########
    # Helpers #
    ###########

    def load_database(self):
        """ Loads json source as a TinyDB database, configured to cache queries,
            I/O operations, as well as serialise datetimes objects if necessary.
            Subjects are initialised as tables of the database

        Returns:
            database (TinyDB)
        """
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
        serialization.register_serializer(TimeDeltaSerializer(), 'TinyDelta')

        database = TinyDB(
            path=self.db_path, 
            sort_keys=True,
            indent=4,
            separators=(',', ': '),
            storage=CachingMiddleware(serialization)
        )

        database.table_class = SmartCacheTable

        return database

    ##################
    # Core Functions #
    ##################

    def create(self, subject, key, new_record):
        """ Creates a new record in a specified subject table within database

        Args:  
            subject (str): Table to be operated on
            new_record (dict): Information for creating a new record
            key (str): Primary key of the current table
        Returns:
            New record added (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            with transaction(subject_table) as tr:

                # Remove additional digits (eg. microseconds)
                date_created = datetime.strptime(
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "%Y-%m-%d %H:%M:%S"
                )
                new_record['created_at'] = date_created

                if subject_table.contains(where(key) == new_record[key]):
                    tr.update(new_record, where(key) == new_record[key])

                else:
                    tr.insert(new_record)

            record = subject_table.get(where(key) == new_record[key])

        return record

    def read_all(self, subject, filter={}):
        """ Retrieves entire collection of records, with an option to filter out
            ones with specific key-value pairs.

        Args:
            filter (dict(str,str)): Key-value pairs for filtering records
        Returns:
            Filtered records (list(tinydb.database.Document))
        """

        def retrieve_all_records(subject):
            """ Retrieves all records in a specified table of the database

            Args:
                subject (str): Table to be operated on
            Returns:
                Records (list(tinydb.database.Document))
            """
            database = self.load_database()

            with database as db:
                subject_table = db.table(subject)
                records = subject_table.all()
                print(records)

            return records

        all_records = retrieve_all_records(subject=subject)
        filtered_records = []
        for record in all_records:
            if (
                (not filter.items() <= record['key'].items()) and
                (not filter.items() <= record.items())
            ):
                continue
            filtered_records.append(record)
        return filtered_records

    def read(self, subject, key, r_id):
        """ Retrieves a single record from a specified table in the database

        Args:  
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (dict): Identifier of specified records
        Returns:
            Specified record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:
            subject_table = db.table(subject)
            record = subject_table.get(where(key) == r_id)

        return record

    def update(self, subject, key, r_id, updates):
        """ Updates an existing record with specified updates

        Args:  
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (dict): Identifier of specified records
            updates (dict): New key-value pairs to update existing record with
        Returns:
            Updated record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            with transaction(subject_table) as tr:

                tr.update(updates, where(key) == r_id)

            updated_record = subject_table.get(where(key) == r_id)

        return updated_record
        
    def delete(self, subject, key, r_id):
        """ Deletes a specified record from the specified table in the database

        Args:
            subject (str): Table to be operated on
            key (str): Primary key of the current table
            r_id (dict): Identifier of specified records
        Returns:
            Deleted record (tinydb.database.Document)
        """
        database = self.load_database()

        with database as db:

            subject_table = db.table(subject)

            record = subject_table.get(where(key) == r_id)

            with transaction(subject_table) as tr:

                tr.remove(where(key) == r_id)
            
            assert not subject_table.get(where(key) == r_id)
        
        return record

####################################
# Data Storage Class - MetaRecords #
####################################

class MetaRecords(Records):
    
    def __init__(self, db_path=db_path):
        super().__init__(db_path=db_path)

    def __generate_key(self, project_id):
        return {"project_id": project_id}

    def create(self, project_id, details):
        # Check that new details specified conforms to export schema 
        # json describing schema is located in synergos_worker/templates/meta_schema.json
        jsonschema.validate(details, schemas["meta_schema"])
        meta_key = self.__generate_key(project_id)
        new_metadata = {'key': meta_key}
        new_metadata.update(details)
        return super().create('Metadata', 'key', new_metadata)

    def read_all(self, filter={}):
        return super().read_all('Metadata', filter=filter)

    def read(self, project_id):
        meta_key = self.__generate_key(project_id)
        return super().read('Metadata', 'key', meta_key)

    def update(self, project_id, updates):
        meta_key = self.__generate_key(project_id)
        return super().update('Metadata', 'key', meta_key, updates)

    def delete(self, project_id):
        meta_key = self.__generate_key(project_id)
        return super().delete('Metadata', 'key', meta_key)

####################################
# Benchmarking Class - Benchmarker #
####################################

class Benchmarker:
    """ Automates the calculation of all supported descriptive statistics

    Attributes:
        y_true (np.ndarray): Truth labels loaded into WSSW
        y_pred (np.ndarray): Predictions obtained from TTP, casted into classes
        y_score (np.ndarray): Raw scores/probabilities obtained from TTP
    """
    def __init__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_score: np.ndarray
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score

    ############
    # Checkers #
    ############

    def is_multiclass(self):
        """ Checks if the current experiment to be evaluated is from a binary or
            multiclass setup

        Returns
            True    if setup is multiclass
            False   otherwise
        """
        try:
            return self.y_score.shape[1] > 1
        except IndexError:
            return False

    ###########
    # Helpers #
    ###########

    @staticmethod
    def _calculate_summary_stats(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, List[Union[int, float]]]:
        """ Given y_true, y_pred & y_score from a classification machine 
            learning operation, calculate the corresponding summary statistics.
        
        Args:
            y_true (np.ndarray)
            y_pred (np.ndarray)
            y_score (np.ndarray)
        Returns:
            Summary Statistics (dict(str, list(int)))
        """
        # Calculate accuracy of predictions
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate ROC-AUC for each label
        try:
            roc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
        except ValueError:
            roc = 0.0
            fpr, tpr = (None, None)
                
        # Calculate Area under PR curve
        pc_vals, rc_vals, _ = precision_recall_curve(y_true, y_score)
        auc_pr_score = auc(rc_vals, pc_vals)
        
        # Calculate F-score
        f_score = f1_score(y_true, y_pred)

        # Calculate contingency matrix
        ct_matrix = contingency_matrix(y_true, y_pred)
        
        statistics = {
            'accuracy': float(accuracy),
            'roc_auc_score': float(roc),
            'pr_auc_score': float(auc_pr_score),
            'f_score': float(f_score)
        }

        plots = {'roc_curve': [fpr, tpr], 'pr_curve': [pc_vals, rc_vals]}

        return statistics
          

    @staticmethod
    def _calculate_descriptive_rates(
        TNs: List[int], 
        FPs: List[int], 
        FNs: List[int], 
        TPs: List[int]
    ) -> Dict[str, List[Union[int, float]]]:
        """ Calculates the descriptive rates for each class in a multiclass 
            setup. Supported rates are as follows:
            1. TPRs: True positive rate
            2. TNRs: True negative rate
            3. PPVs: Positive predictive value
            4. NPVs: Negative predictive value
            5. FPRs: False positive rate
            6. FNRs: False negative rate
            7. FDRs: False discovery rate

        Args:
            TNs (list(float)): No. of true negatives for all classes
            FPs (list(float)): No. of false positives for all classes
            FNs (list(float)): No. of false negatives for all classes
            TPs (list(float)): No. of true positives for all classes
        Returns:
            Descriptive Rates (dict(str, list(float)))
        """
        rates = {}

        def add_rate(r_type, value):
            target_rates = rates.get(r_type, [])
            target_rates.append(value)
            rates[r_type] = [float(value) for value in target_rates]

        for TN, FP, FN, TP in zip(TNs, FPs, FNs, TPs):

            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
            add_rate('TPRs', TPR)

            # Specificity or true negative rate
            TNR = TN/(TN+FP) if (TN+FP) != 0 else 0
            add_rate('TNRs', TNR)

            # Precision or positive predictive value
            PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
            add_rate('PPVs', PPV)

            # Negative predictive value
            NPV = TN/(TN+FN) if (TN+FN) != 0 else 0
            add_rate('NPVs', NPV)

            # Fall out or false positive rate
            FPR = FP/(FP+TN) if (FP+TN) != 0 else 0
            add_rate('FPRs', FPR)

            # False negative rate
            FNR = FN/(TP+FN) if (TP+FN) != 0 else 0
            add_rate('FNRs', FNR)

            # False discovery rate
            FDR = FP/(TP+FP) if (TP+FP) != 0 else 0
            add_rate('FDRs', FDR)

        return rates


    def _find_stratified_descriptors(self) -> Dict[str, List[Union[int, float]]]:
        """ Finds the values of descriptors for all classes in a multiclass
            setup. Descriptors are True Negatives (TNs), False Positives (FPs),
            False Negatives (FNs) and True Positives (TPs).

        Returns:
            Stratified Descriptors (dict(str, list(int)))
        """
        # Calculate confusion matrix
        cf_matrix = confusion_matrix(self.y_true, self.y_pred)
        logging.debug(f"Confusion matrix: {cf_matrix}")

        FPs = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
        FNs = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
        TPs = np.diag(cf_matrix)
        TNs = cf_matrix[:].sum() - (FPs + FNs + TPs)
        logging.debug(f"TNs: {TNs}, FPs: {FPs}, FNs: {FNs}, TP: {TPs}")
        
        descriptors = {'TNs': TNs, 'FPs': FPs, 'FNs': FNs, 'TPs': TPs}
        for des_type, descriptor_values in descriptors.items():
            descriptors[des_type] = [
                int(value) for value in descriptor_values
            ]
        return descriptors


    def _calculate_stratified_stats(self) -> Dict[str, List[Union[int, float]]]:
        """ Calculates descriptive statistics of a classification run. 
            Statistics supported are accuracy, roc_auc_score, pr_auc_score and
            f_score

        Returns:
            Stratified Statistics (dict(str, list(float)))
        """
        ohe_y_true, ohe_y_pred, ohe_y_score= self._decode_ohe_dataset()

        aggregated_statistics = {}
        for col_true, col_pred, col_score in zip(
            ohe_y_true.T, 
            ohe_y_pred.T, 
            ohe_y_score.T
        ):
            label_statistics = self._calculate_summary_stats(
                y_true=col_true, 
                y_pred=col_pred, 
                y_score=col_score
            )
            for metric, value in label_statistics.items():
                metric_collection = aggregated_statistics.get(metric, [])
                metric_collection.append(value)
                aggregated_statistics[metric] = metric_collection

        return aggregated_statistics


    def _analyse_regression(self) -> Dict[str, float]:
        """ Automates calculation of descriptive statistics, assuming that 
            y_true, y_pred & y_scores all correspond to a regression machine
            learning operation.
                  
            Statistics supported for regression include:
            1) R2   : R-squared
            2) MSE  : Mean Squared Error
            3) MAE  : Mean Absolute Error

        Returns:
            Regression statistics (dict)
        """
        R2 = r2_score(self.y_true, self.y_pred)
        MSE = mean_squared_error(self.y_true, self.y_pred)
        MAE = mean_absolute_error(self.y_true, self.y_pred)
        return {'R2': R2, 'MSE': MSE, 'MAE': MAE}


    def _analyse_classification(self) -> Dict[str, List[Union[int, float]]]:
        """ Automates calculation of descriptive statistics, assuming that 
            y_true, y_pred & y_scores all correspond to a classification
            machine learning operation.

            Statistics supported for classification include:
            1) accuracy
            2) roc_auc_score
            3) pr_auc_score
            4) f_score
            5) TPRs
            6) TNRs
            7) PPVs
            8) NPVs
            9) FPRs
            10) FNRs
            11) FDRs
            12) TPs
            13) TNs
            14) FPs
            15) FNs

        Returns:
            Classification statistics (dict)
        """
        statistics = self._calculate_stratified_stats()
        descriptors = self._find_stratified_descriptors()
        rates = self._calculate_descriptive_rates(**descriptors)
        statistics.update(descriptors)
        statistics.update(rates)

        return statistics


    def _decode_ohe_dataset(self):
        """ Reverses one-hot encoding applied on a dataset, while maintaining 
            the original class representations
            Assumption: This function can only be used if action is to classify
        """
        if self.is_multiclass():
            ohe_y_true = th.nn.functional.one_hot(
                th.as_tensor(self.y_true),
                num_classes=self.y_score.shape[-1]
            ).numpy()
            ohe_y_pred = th.nn.functional.one_hot(
                th.as_tensor(self.y_pred),
                num_classes=self.y_score.shape[-1]
            ).numpy()
            ohe_y_score = self.y_score

        else:
            ohe_y_true = np.concatenate((1-self.y_true, self.y_true), axis=1)
            ohe_y_pred = np.concatenate((1-self.y_pred, self.y_pred), axis=1)
            ohe_y_score = np.concatenate((1-self.y_score, self.y_score), axis=1)

        logging.debug(f"OHE y_true: {ohe_y_true}")
        logging.debug(f"OHE y_pred: {ohe_y_pred}")
        logging.debug(f"OHE y_score: {ohe_y_score}")
        return ohe_y_true, ohe_y_pred, ohe_y_score

    def reconstruct_dataset(self):
        """ Searches WebsocketServerWorker for dataset objects and their
            corresponding predictions, before stitching them back into a 
            single dataframe.
        """
        raise NotImplementedError

    ##################
    # Core Functions #
    ##################

    def reconstruct(self):
        """ Given a mapping of dataset object IDs to their respective prediction
            object IDs, reconstruct an aggregated dataset with predictions
            mapped for client's perusal
        """
        raise NotImplementedError


    def analyse(self, action: str) -> dict:
        """ Automates calculation of descriptive statistics over restored 
            batched data. 
            
        Args:
            action (str): Type of ML operation to be executed. Supported options
                are as follows:
                1) 'regress': Orchestrates FL grid to perform regression
                2) 'classify': Orchestrates FL grid to perform classification
                3) 'cluster': TBA
                4) 'associate': TBA
        Returns:
            Statistics (dict)
        """
        if action == "regress":
            return self._analyse_regression()

        elif action == "classify":
            return self._analyse_classification()

        else:
            raise ValueError(f"ML action {action} is not supported!")
        

    def export(self, out_dir):
        """ Exports reconstructed dataset to file for client's perusal
        """
        raise NotImplementedError

#######################################
# MetaExtractor Class - MetaExtractor #
#######################################

class MetaExtractor:
    """ Given a dataset of a specific type, extract the appropriate meta
        statistics for ease of summary

    Attributes:
        df (pd.DataFrame): Dataset to extract metrics from 
        schema
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        schema: Dict[str, str], 
        dataset_type: str
    ):
        # Private attibutes
        self.metadata = None
        
        # Public Attributes
        self.df = df
        self.schema = schema
        self.dataset_type = dataset_type

    ###########
    # Helpers #
    ###########

    @staticmethod
    def extract_categorical_feature_metadata(
        feature: pd.Series
    ) -> Dict[str, Union[List[str], int, float]]:
        """ Extracts relevant statistics from a single categorical feature.
            For categorical variables, supported metadata extracted include:
            1) Labels
            2) Count
            3) Unique
            4) Top
            5) Frequency 

        Args:
            feature (pd.Series): Name of feature column
        Returns:
            Categorical Meta statistics (Dict)
        """
        datatype = feature.dtype.name

        if datatype == "category":
            
            # Extract class labels
            labels = feature.cat.categories.to_list()
            logging.debug(f"------> {[type(v) for v in labels]}")

            # Extract meta statistics
            meta_stats = feature.describe().to_dict()
            logging.debug(f"------> {[type(v) for k,v in meta_stats.items()]}")

            # Add in class label information
            meta_stats.update({'labels': labels})
            return meta_stats

        else:
            raise RuntimeError(f"Feature '{feature.name}' is not a categorical variable!")


    @staticmethod
    def extract_numeric_feature_metadata(
        feature: pd.Series
    ) -> Dict[str, Union[int, float]]:
        """ Extracts relevant statistics from a single numeric feature.
            For numeric variables, supported metadata extracted include:
            1) Count
            2) Mean
            3) Std
            4) Min
            5) 25% 
            6) 50% 
            7) 75% 
            8) max 

        Args:
            feature (pd.Series): Name of feature column
        Returns:
            Numerical Meta statistics (Dict)
        """
        datatype = feature.dtype.name
        
        if datatype not in ["category", "object"]: # capture nulls
            logging.debug(f"------> {[type(v) for k,v in feature.describe().to_dict().items()]}")

            return feature.describe().to_dict()

        else:
            raise RuntimeError(f"Feature '{name}' is not a numerical variable!")


    @staticmethod
    def extract_object_feature_metadata(feature: pd.Series) -> dict:
        """ Extracts relevant statistics from a single object feature. 

            Note: 
            This is a placeholder function to handle nullities/incompatibilities
            in the event that the specified dataset was not thoroughly cleaned

        Args:
            feature (pd.Series): Name of feature column
        Returns:
            An empty dictionary (Dict)
        """
        return {}


    def extract_tabular_metadata(
        self
    ) -> Dict[str, Dict[str, Union[List[str], int, float]]]:
        """ Extracts meta data/statistics from a specified tabular dataset.
            Expected metadata format:
            {
                'features': {
                    'cat_variables': {
                        'cat_feature_1': {'datatype': "category", ...},
                        ...
                    },
                    'num_variables': {
                        'num_feature_1': {'datatype': "integer", ...},
                        ...
                    },
                    'misc_variables': {
                        'misc_feature_1': {'datatype': "object"},
                        ...
                    }
                }
            }

        Returns:
            Tabular meta statistics (Dict)
        """
        if self.dataset_type == "tabular":

            # Ensures that template always has consistent keys
            metadata = {
                'cat_variables': {},
                'num_variables': {},
                'misc_variables': {}
            }
            for name in self.df.columns:

                feature = self.df[name]
                datatype = self.schema[name] # robust datatype extraction

                # Check that datatype is not ambigious (eg. null, list, etc.)
                if datatype == "object":
                    variable_key = 'misc_variables'
                    meta_stats = self.extract_object_feature_metadata(feature)

                # Check that datatype is categorical
                elif datatype == "category":
                    variable_key = 'cat_variables'
                    meta_stats = self.extract_categorical_feature_metadata(feature)

                # Check that datatype is numerical
                else:
                    ###########################
                    # Implementation Footnote #
                    ###########################
                    
                    # [Cause]
                    # There are many other datatypes apart from objects & 
                    # categories in numpy.

                    # [Problems]
                    # This results in ambiguity when inferring numeric datatypes

                    # [Solution]
                    # Assume that all boolean types are specified as categories

                    variable_key = 'num_variables'
                    meta_stats = self.extract_numeric_feature_metadata(feature)

                meta_stats['datatype'] = datatype

                variable_stats = metadata.get(variable_key, {})
                variable_stats[name] = meta_stats
                metadata.update({variable_key: variable_stats})
            
            return {'features': metadata}
        
        else:
            raise RuntimeError(f"Dataset is not of type tabular!")


    def extract_image_metadata(self) -> Dict[str, Union[int, str]]:
        """ Extracts meta data/statistics from a specified tabular dataset.
            Expected metadata format:
            {
                'pixel_height': 255,
                'pixel_width': 255,
                'color': "rgb", # for now, only grayscale & RGB is supported
            }

        Returns:
            Image meta statistics
        """
        if self.dataset_type == "image":

            # Columns of image DFs are named "{img_format}x{height}x{width}"
            features = self.df.drop(columns=['target'])
            color, pixel_height, pixel_width = features.columns[-1].split('x')

            return {
                'pixel_height': int(pixel_height),
                'pixel_width': int(pixel_width),
                'color': color
            }

        else:
            raise RuntimeError(f"Dataset is not of type image!")

    
    def extract_text_metadata(self) -> Dict[str, Union[int, float]]:
        """ Extracts meta data/statistics from a specified text dataset. 
        
            Assumption:
            Text datasets are represented as doc-term matrices

            Expected metadata format:
            {
                'word_count': 5000,     # Total no. of words represented
                'sparsity': 0.6         # count(zeros)/total of doc-term matrix
                'representation': 0.2   # sum(non-zeros)/total of doc-term matrix
            }

        Returns:
            Text meta statistics (Dict)
        """
        if self.dataset_type == "text":

            features = self.df.drop(columns=['target'])

            doc_count, word_count = features.shape

            total_cells = doc_count * word_count

            zero_count = features[features==0].count().sum()
            sparsity = zero_count/total_cells

            non_zero_sum = features.sum().sum()  # .sum().sum() bypasses nullity
            representation = non_zero_sum/total_cells

            return {
                'word_count': word_count,
                'sparsity': sparsity,
                'representation': representation
            }

        else:
            raise RuntimeError(f"Dataset is not of type text!")


    def extract_generic_metadata(self) -> Dict[str, Union[int, str]]:
        """ Extracts generic meta data/statistics of the specified dataset.

        Returns:
            Generic meta statistics (Dict)
        """
        return {
            'src_count': len(self.df),
            '_type': self.dataset_type
        }

    ##################
    # Core functions #
    ##################

    def extract(self) -> Dict[str, Union[str, int, float, dict]]:
        """ Extracts & compiles all metadata for each feature within the 
            specified dataset.

            Expected metadata format:
            {
                'src_count': 1000,
                '_type': "<insert datatype>",
                <insert type-specific meta statistics>
                ...
            }

        Returns:
            Data-specific meta statistics (Dict)
        """
        EXTRACTORS = {
            'tabular': self.extract_tabular_metadata,
            'image': self.extract_image_metadata,
            'text': self.extract_text_metadata
        }
        supported_dataset_types = list(EXTRACTORS.keys())

        if self.dataset_type not in supported_dataset_types:
            raise RuntimeError(f"{self.dataset_type} is not yet supported!")
            
        generic_metadata = self.extract_generic_metadata()
        type_specific_metadata = EXTRACTORS[self.dataset_type]()

        self.metadata = {**generic_metadata, **type_specific_metadata}
        return self.metadata
