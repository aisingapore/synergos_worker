#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict

# Libs
import numpy as np
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.core.server import load_metadata_records, load_proc
from rest_rpc.core.utils import Payload

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "poll", 
    description='API to faciliate metadata retrieval from participant.'
)

poll_queue = mp.Queue() # Method 1: Multiprocessing
# poll_queue = Queue()    # Method 2: Multi-threading

logging = app.config['NODE_LOGGER'].synlog
logging.debug("poll.py logged", Description="No Changes")

#############
# Functions #
#############

def run_archival_jobs(job_queue):
    """ Extracts job from polling queue and performs them sequentially. This is 
        used as a safety feature for controlling the number of polling 
        operations active at anytime, so as to prevent crashes during different
        phases of the federated cycle.

    Args:
        job_queue (): Intermediary queue that steamlines the number of loading
            jobs operable on the worker at any time.
    """

    def build_archives(keys: Dict[str, str], **kwargs):
        """ Constructs archive based on specified datasets and caches them in 
            the operations database for future use. 

        Args:
            keys (dict(str,str)): All prerequisite IDs to build a project archive
            **kwargs: All parameters required to trigger archive building.
                Current parameters include:
                1) Action    --> What kind of ML operation to be done
                2) Data tags --> Tokens to location of data sources
        Returns:
            Core archive (i.e. no relations attached) (dict)
        """
        meta_records = load_metadata_records(keys=keys)
        operation_archive = load_proc(keys=keys, **kwargs)

        meta_id = keys.get('project_id')
        updated_entry = meta_records.update(meta_id, updates=operation_archive)

        return updated_entry

    while True:
        logging.debug(
            "Polling job queue is live, awaiting jobs...",
            ID_path=SOURCE_FILE,
            ID_function=run_archival_jobs.__name__
        )

        while not job_queue.empty():
            logging.info(
                f"No. of jobs in queue: {job_queue.qsize()}",
                job_count=job_queue.qsize(),
                ID_path=SOURCE_FILE,
                ID_function=run_archival_jobs.__name__
            )

            parameters = job_queue.get() # blocks until queue is populated
            keys = parameters['keys']
            logging.info(
                f"Archival job for entry key {keys} received!",
                keys=keys,
                ID_path=SOURCE_FILE,
                ID_function=run_archival_jobs.__name__
            )

            # try:
            created_archive = build_archives(**parameters)
            logging.info(
                f"Archive created for entry key {keys}!",
                keys=keys,
                archive=created_archive,
                ID_path=SOURCE_FILE,
                ID_function=run_archival_jobs.__name__
            )

            # except Exception as e:
            #     logging.error(
            #         f"Archive for entry key {keys} failed to build! Error: {e}",
            #         keys=keys,
            #         ID_path=SOURCE_FILE,
            #         ID_function=run_archival_jobs.__name__
            #     )
            #     pass

            # job_queue.task_done()

        import time
        time.sleep(1)

### Method 1: Multiprocessing ####

# Internal queuing mechanism to prevent system overload
archival_process = mp.Process(target=run_archival_jobs, args=(poll_queue,))

# Ensures that when process exits, it attempts to terminate all of its 
# daemonic child processes.
archival_process.daemon = True 

archival_process.start()


### Method 2: Multi-threading ####

# archival_thread = Thread(target=run_archival_jobs, args=(poll_queue,))
# archival_thread.daemon = True
# archival_thread.start()

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling Inputs
tag_model = ns_api.model(
    name="tags",
    model={
        'train': fields.List(fields.List(fields.String()), required=True),
        'evaluate': fields.List(fields.List(fields.String())),
        'predict': fields.List(fields.List(fields.String())),
        'model': fields.List(fields.String()),
        'hyperparameters': fields.List(fields.String())
    }
)

poll_input_model = ns_api.model(
    name="poll_input",
    model={
        'action': fields.String(),
        'tags': fields.Nested(tag_model, required=True)
    }
)

# Marshalling outputs
xy_sequence_model = ns_api.model(
    name="xy_sequences",
    model={
        'X': fields.List(fields.String(), required=True),
        'y': fields.List(fields.String(), required=True)
    }
)

header_model = ns_api.model(
    name="headers",
    model={
        'train': fields.Nested(xy_sequence_model, required=True),
        'evaluate': fields.Nested(xy_sequence_model, skip_none=True),
        'predict': fields.Nested(xy_sequence_model, skip_none=True)
    },
    skip_none=True
)

schema_field = fields.Wildcard(fields.String())
meta_model = ns_api.model(
    name="meta_schema",
    model={"*": schema_field}   # eg. {'x1': "float64", 'x2': "int64", ...}
)

schema_model = ns_api.model(
    name="schema",
    model={
        'train': fields.Nested(meta_model, required=True),
        'evaluate': fields.Nested(meta_model, skip_none=True),
        'predict': fields.Nested(meta_model, skip_none=True)
    }
)

generic_feature_model = ns_api.model(
    name="generic_feature_stats",
    model={'datatype': fields.String(required=True)}
)

categorical_feature_model = ns_api.inherit(
    "categorical_feature_metadata",
    generic_feature_model,
    {
        'labels': fields.List(fields.String(), required=True),
        'count': fields.Integer(required=True),
        'unique': fields.Integer(required=True),
        'top': fields.String(required=True),
        'freq': fields.Integer(required=True)
    }
)
categorical_meta_field = fields.Wildcard(fields.Nested(categorical_feature_model))
categorical_meta_model = ns_api.model(
    name="categorical_metadata",
    model={"*": categorical_meta_field}
)

numeric_feature_model = ns_api.inherit(
    "numeric_feature_metadata",
    generic_feature_model,
    {
        'count': fields.Float(required=True),
        'mean': fields.Float(required=True),
        'std': fields.Float(required=True),
        'min': fields.Float(required=True),
        '25%': fields.Float(required=True),
        '50%': fields.Float(required=True),
        '75%': fields.Float(required=True),
        'max': fields.Float(required=True)
    }
)
numeric_meta_field = fields.Wildcard(fields.Nested(numeric_feature_model))
numeric_meta_model = ns_api.model(
    name="numeric_metadata",
    model={"*": numeric_meta_field}
)

misc_meta_field = fields.Wildcard(fields.Nested(generic_feature_model))
misc_meta_model = ns_api.model(
    name="misc_metadata",
    model={"*": misc_meta_field} # No marshalling enforced
)

feature_summary_meta_model = ns_api.model(
    name="feature_summary_metadata",
    model={
        'cat_variables': fields.Nested(
            categorical_meta_model, required=True, skip_none=True, 
        ),
        'num_variables': fields.Nested(
            numeric_meta_model, required=True, skip_none=True,
        ),
        'misc_variables': fields.Nested(
            misc_meta_model, required=True, skip_none=True,
        )
    }
)

tabular_meta_model = ns_api.model(
    name="tabular_metadata",
    model={
        'features': fields.Nested(
            feature_summary_meta_model, 
            required=True, 
            default={}
        )
    }
)

image_meta_model = ns_api.model(
    name="image_metadata",
    model={
        'pixel_height': fields.Integer(required=True),
        'pixel_width': fields.Integer(required=True),
        'color': fields.String(required=True)
    }
)

text_meta_model = ns_api.model(
    name="text_metadata",
    model={
        'word_count': fields.Integer(required=True),
        'sparsity': fields.Float(required=True),
        'representation': fields.Float(required=True)
    }
)

generic_meta_model = ns_api.model(
    name="generic_metadata",
    model={
        'src_count': fields.Integer(required=True),
        '_type': fields.String(required=True)
    }
)

dataset_meta_model = ns_api.inherit(
    "dataset_metadata", 
    generic_meta_model,
    tabular_meta_model,
    image_meta_model,
    text_meta_model
)

metadata_model = ns_api.model(
    name="metadata",
    model={
        'train': fields.Nested(dataset_meta_model, required=True, skip_none=True),
        'evaluate': fields.Nested(dataset_meta_model, skip_none=True),
        'predict': fields.Nested(dataset_meta_model, skip_none=True)
    }
)

poll_model = ns_api.model(
    name="poll",
    model={
        'headers': fields.Nested(header_model, required=True),
        'schemas': fields.Nested(schema_model, required=True),
        'metadata': fields.Nested(metadata_model, required=True),
        # Exports will not be made available to the TTP 
    }
)

poll_output_model = ns_api.inherit(
    "poll_output",
    poll_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String(),
                    'project_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = Payload('Poll', ns_api, poll_output_model)

#############
# Resources #
#############

@ns_api.route('/<collab_id>/<project_id>')
@ns_api.response(200, "Initialised project logs successfully")
@ns_api.response(417, "Insufficient info specified for metadata tracing")
@ns_api.response(500, "Internal failure")
class Poll(Resource):
    """ Provides necessary information from participant for orchestration """

    @ns_api.doc("poll_retrieve_metadata")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id, project_id):
        """ Retrieves specified metadata regarding the worker.

            JSON sent will contain the following information:
            1) Data Headers

            eg.

            {
                "headers": {
                    "train": {
                        "X": ["X1_1", "X1_2", "X2_1", "X2_2", "X3"],
                        "y": ["target_1", "target_2"]
                    },
                    "evaluate": {
                        "X": ["X1_1", "X1_2", "X2_1", "X3"],
                        "y": ["target_1", "target_2"]
                    }
                },
                "schemas": {
                    "train": {
                        "X1": "int32",
                        "X2": "category", 
                        "X3": "category", 
                        "X4": "int32", 
                        "X5": "int32", 
                        "X6": "category", 
                        "target": "category"
                    },
                    ...
                },
                "metadata":{
                    "train":{
                        'src_count': 1000,
                        '_type': "<insert datatype>",
                        <insert type-specific meta statistics>
                        ...
                    },
                    ...
                }
            }
        """
        # Search local database for cached operations
        meta_records = load_metadata_records(keys=request.view_args)
        retrieved_metadata = meta_records.read(project_id)

        logging.debug(
            "Retrieved metadata from database tracked.",
            retrieved_metadata=retrieved_metadata,
            ID_path=SOURCE_FILE,
            ID_class=Poll.__name__, 
            ID_function=Poll.get.__name__,
            **request.view_args
        )
    
        logging.debug(
            "Tags received from TTP tracked.",
            received_tags=request.json['tags'],
            ID_path=SOURCE_FILE,
            ID_class=Poll.__name__, 
            ID_function=Poll.get.__name__,
            **request.view_args
        )

        if retrieved_metadata:

            # Check if important information (i.e. headers, schemas & metadata)
            # have already been injected into the archive records, AND that
            # there has been no changes to the declared dataset sets (via tags)
            if (
                retrieved_metadata['tags'] and 
                retrieved_metadata['headers'] and 
                retrieved_metadata['schemas'] and
                retrieved_metadata['metadata'] and
                retrieved_metadata['exports']
            ) and (
                retrieved_metadata['tags'] == request.json['tags']
            ):
                success_payload = payload_formatter.construct_success_payload(
                    status=200,
                    method="poll.get",
                    params=request.view_args,
                    data=retrieved_metadata
                )

                logging.info(
                    "State polling retrieval successfully completed!", 
                    code="200", 
                    ID_path=SOURCE_FILE,
                    ID_class=Poll.__name__, 
                    ID_function=Poll.get.__name__,
                    **request.view_args
                )

                return success_payload, 200

            # Otherwise, this means that the archival job corresponding to the
            # entry key is either still enqueued, or in progress. Either way, it
            # means that the record is not ready for use.
            else:
                ns_api.abort(
                    code=406, 
                    message=f"Archival job for project '{project_id}' is still in progress! Please try again later."
                )

        else:
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' does not exist!"
            )


    @ns_api.doc("poll_load_metadata")
    @ns_api.expect(poll_input_model)
    # @ns_api.marshal_with(payload_formatter.singular_model)
    def post(self, collab_id, project_id):
        """ Retrieves specified metadata regarding the worker.

            JSON received will contain the following information:
            1) Action (i.e. 'regress', 'classify', 'cluster', 'associate')
            2) Connections
            3) Data tags

            eg. 

            {
                "action": 'classify',
                "connections": {
                    'logs': {
                        'host': "172.18.0.4",
                        'port': 5000,
                        'configurations': {
                            name: "test_participant_1",
                            logging_level: 20,
                            logging_variant: "graylog",
                            debugging_fields: False,
                        }
                    }
                },
                "tags": {
                    "train": [["type_a","v1"], ["type_b","v2"]],
                    "evaluate": [["type_c","v3"]]
                }
            }

            JSON sent will contain the following information:
            1) Data Headers

            eg.

            {
                "headers": {
                    "train": {
                        "X": ["X1_1", "X1_2", "X2_1", "X2_2", "X3"],
                        "y": ["target_1", "target_2"]
                    },
                    "evaluate": {
                        "X": ["X1_1", "X1_2", "X2_1", "X3"],
                        "y": ["target_1", "target_2"]
                    }
                },
                "schemas": {
                    "train": {
                        "X1": "int32",
                        "X2": "category", 
                        "X3": "category", 
                        "X4": "int32", 
                        "X5": "int32", 
                        "X6": "category", 
                        "target": "category"
                    }
                    ...
                },
                "metadata":{
                    "train":{
                        'src_count': 1000,
                        '_type': "<insert datatype>",
                        <insert type-specific meta statistics>
                        ...
                    },
                    ...
                }
            }
        """
        # Search local database for cached operations
        meta_records = load_metadata_records(keys=request.view_args)
        retrieved_metadata = meta_records.read(project_id)

        # Polling initialises the project logs if it does not exist yet
        if not retrieved_metadata:
            meta_records.create(
                project_id=project_id, 
                details={
                    'tags': {},
                    'headers': {},
                    'schemas': {},
                    'metadata': {},
                    'exports': {},
                    'process': None,    # process ID hosting WSSW
                    'is_live': False,   # state of WSSW
                    'in_progress': [],  # cycle combination(s) queued
                    'connections': [],
                    'results': {}
                }
            )
            retrieved_metadata = meta_records.read(project_id)

        # If polling operation had already been done before, skip preprocessing
        # (Note: this is only valid if the submitted set of tags are the same)
        if retrieved_metadata['tags'] == request.json['tags']:
            status = 201    # resource was already created

        # Otherwise, submit a job to the internal poll queue to perform 
        # preprocessing, and archive results of operation
        else:
            parameters = {'keys':request.view_args, **request.json}
            poll_queue.put_nowait(parameters)
            status = 202    # Job to create resource has been accepted
            
        # data = {'jobs': list(poll_queue.queue)} # return all jobs in queue
        data = {'jobs': poll_queue.qsize()} # return all jobs in queue

        success_payload = payload_formatter.construct_success_payload(
            status=status,
            method="poll.post",
            params=request.view_args,
            data=data,
            strict_format=False
        )
        return success_payload, status
