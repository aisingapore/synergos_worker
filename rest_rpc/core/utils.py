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
from flask import jsonify, request
from flask_restx import fields
from sklearn.metrics import (
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
            return (
                self.y_true.shape[1] > 1 and 
                self.y_pred.shape[1] > 1 and  
                self.y_score.shape[1] > 1
            )
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
        """

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


    def _find_stratified_descriptors(
        self
    ) -> Dict[str, List[Union[int, float]]]:
        """ Finds the values of descriptors for all classes in a multiclass
            setup. Descriptors are True Negatives (TNs), False Positives (FPs),
            False Negatives (FNs) and True Positives (TPs).

        Returns:
            Stratified Descriptors (dict(str, list(int)))
        """
        # Calculate confusion matrix
        cf_matrix = confusion_matrix(
            np.argmax(self.y_true, axis=1) if self.is_multiclass() else self.y_true,
            np.argmax(self.y_pred, axis=1) if self.is_multiclass() else self.y_pred
        )
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


    def _calculate_stratified_stats(
        self
    ) -> Dict[str, List[Union[int, float]]]:
        """ Calculates descriptive statistics of a training run. Statistics 
            supported are accuracy, roc_auc_score, pr_auc_score and f_score

        Returns:
            Stratified Statistics (dict(str, list(float)))
        """
        aggregated_statistics = {}
        for col_true, col_pred, col_score in zip(
            self.y_true.T, 
            self.y_pred.T, 
            self.y_score.T
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


    def decode_ohe_dataset(self, dataset, header, alignment):
        """ Reverses one-hot encoding applied on a dataset
        """
        raise NotImplementedError


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


    def analyse(self):
        """ Automates calculation of descriptive statistics over restored 
            batched data. 
            
            Statistics supported include:
            1) accuracy,
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
            Statistics (dict)
        """
        statistics = self._calculate_stratified_stats()

        descriptors = self._find_stratified_descriptors()
        rates = self._calculate_descriptive_rates(**descriptors)
        statistics.update(descriptors)
        statistics.update(rates)

        return statistics
        


    def export(self, out_dir):
        """ Exports reconstructed dataset to file for client's perusal
        """
        raise NotImplementedError