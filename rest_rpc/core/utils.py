#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import uuid
from datetime import datetime

# Libs
import jsonschema
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
        y_true (np.array): Truth labels loaded into WSSW
        y_pred (np.array): Predictions obtained from TTP, casted into classes
        y_score (np.array): Raw scores/probabilities obtained from TTP
    """
    def __init__(self, y_true, y_pred, y_score):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score

    def calculate_stats(self):
        """ Calculates descriptive statistics of a training run. Statistics
            supported include:
            1) accuracy,
            2) roc_auc_score
            3) pr_auc_score
            4) f_score
            5) TPR
            6) TNR
            7) PPV
            8) NPV
            9) FPR
            10) FNR
            11) FDR
            12) TP
            13) TN
            14) FP
            15) FN

        Returns:
            Statistics (dict)
        """
        # Calculate accuracy of predictions
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Calculate ROC-AUC for each label
        roc = roc_auc_score(self.y_true, self.y_score)
        fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
        
        # Calculate Area under PR curve
        pc_vals, rc_vals, _ = precision_recall_curve(self.y_true, self.y_score)
        auc_pr_score = auc(rc_vals, pc_vals)
        
        # Calculate F-score
        f_score = f1_score(self.y_true, self.y_pred)

        # Calculate contingency matrix
        ct_matrix = contingency_matrix(self.y_true, self.y_pred)
        
        # Calculate confusion matrix
        cf_matrix = confusion_matrix(self.y_true, self.y_pred)
        logging.debug(f"Confusion matrix: {cf_matrix}")

        TN, FP, FN, TP = cf_matrix.ravel()
        logging.debug(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
        # Specificity or true negative rate
        TNR = TN/(TN+FP) if (TN+FP) != 0 else 0
        # Precision or positive predictive value
        PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
        # Negative predictive value
        NPV = TN/(TN+FN) if (TN+FN) != 0 else 0
        # Fall out or false positive rate
        FPR = FP/(FP+TN) if (FP+TN) != 0 else 0
        # False negative rate
        FNR = FN/(TP+FN) if (TP+FN) != 0 else 0
        # False discovery rate
        FDR = FP/(TP+FP) if (TP+FP) != 0 else 0

        return {
            'accuracy': float(accuracy),
            'roc_auc_score': float(roc),
            'pr_auc_score': float(auc_pr_score),
            'f_score': float(f_score),
            'TPR': float(TPR),
            'TNR': float(TNR),
            'PPV': float(PPV),
            'NPV': float(NPV),
            'FPR': float(FPR),
            'FNR': float(FNR),
            'FDR': float(FDR),
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        }

    def decode_ohe_dataset(self, dataset, header, alignment):
        """ Reverses one-hot encoding applied on a dataset
        """
        pass

    def reconstruct_dataset(self):
        """ Searches WebsocketServerWorker for dataset objects and their
            corresponding predictions, before stitching them back into a 
            single dataframe.
        """
        pass

    ##################
    # Core Functions #
    ##################

    def reconstruct(self):
        """ Given a mapping of dataset object IDs to their respective prediction
            object IDs, reconstruct an aggregated dataset with predictions
            mapped for client's perusal
        """
        pass


    def analyse(self):
        """ Automates calculation of descriptive statistics over restored 
            batched data
        """
        pass


    def export(self, out_dir):
        """ Exports reconstructed dataset to file for client's perusal
        """
        pass