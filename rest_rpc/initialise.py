#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
from pathlib import Path

# Libs
import jsonschema
import numpy as np
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.core.server import start_proc
from rest_rpc.core.utils import Payload, MetaRecords, construct_combination_key
from rest_rpc.poll import tag_model, schema_model
from rest_rpc.align import alignment_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "initialise", 
    description='API to faciliate WSSW startup for participant.'
)

cache = app.config['CACHE']

db_path = app.config['DB_PATH']
meta_records = MetaRecords(db_path=db_path)

cache_template = app.config['CACHE_TEMPLATE']
outdir_template = cache_template['out_dir']

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling Inputs
server_model = ns_api.model(
    name="server",
    model={
        'id': fields.String(required=True),
        'host': fields.String(required=True),
        'port': fields.Integer(required=True),
        'log_msgs': fields.Boolean(),
        "verbose": fields.Boolean()
    }
)

init_input_model = ns_api.inherit(
    "initialisation_input",
    server_model,
    {
        'tags': fields.Nested(tag_model),
        'alignments': fields.Nested(alignment_model)
    }
)

# Marshalling Outputs
init_output_model = ns_api.model(
    name="initialisation_output",
    model={
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        ),
        'is_live': fields.Boolean(required=True),
        'in_progress': fields.List(fields.String(), required=True),
        'connections': fields.List(fields.String(), required=True)
    }
)

payload_formatter = Payload('Initialise', ns_api, init_output_model)

#############
# Resources #
#############

@ns_api.route('/<project_id>/<expt_id>/<run_id>')
@ns_api.response(200, "Existing WSSW object found")
@ns_api.response(201, "New WSSW object instantiated")
@ns_api.response(404, "Project logs has not been initialised")
@ns_api.response(417, "Insufficient info specified for WSSW initialisation")
@ns_api.response(500, 'Internal failure')
class Initialisation(Resource):
    
    @ns_api.doc("initialise_wssw")
    @ns_api.expect(init_input_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def post(self, project_id, expt_id, run_id):
        """ Start up WebsocketServerWorker for representing participant hosting
            this worker container
            
            JSON received will contain the following information:
            1) Run ID
            2) Worker ID
            3) Host IP
            4) Host Port
            5) Verbosity
            6) Tags
            7) Alignments

            eg.

            {
                "host": "0.0.0.0"
                "port": 8020,
                "id": "worker_0",
                "log_msgs": true,
                "verbose": true,
                "tags": {
                    "train": [["type_a","v2"], ["type_b","v3"]],
                    "evaluate": [["type_c", "v1"]]
                },
                "alignments": {
                    "train": {
                        "X": [0,1,3,6,8],
                        "y": [1]
                    },
                    "evaluate": {
                        "X": [0,1,3,6,8,9],
                        "y": [2],
                    }
                }
            }
        
            JSON sent will contain the following information:
            1) is_live - Checks if an existing instance of WSSW is running
            2) Connections - Concurrent experiment-runs found under project
            3) in_progress - Tracks no. of runs remaining

            eg.

            {
                "is_live": true,
                "in_progress": [
                    "(test_expt_2, test_run_2)",
                    "(test_expt_3, test_run_3)"                    
                ]
                "connections": [
                    "(test_expt_1, test_run_1)",
                    "(test_expt_2, test_run_2)",
                    "(test_expt_3, test_run_3)"
                ]
            }
        """ 
        # Search local database for cached operations
        retrieved_metadata = meta_records.read(project_id)

        if retrieved_metadata:

            # Check that specified experiment run is not already running
            if not cache[project_id]:

                project_cache_dir = os.path.join(
                    outdir_template.safe_substitute(project_id=project_id), 
                    "cache"
                )
                wssw_process, wss_worker = start_proc(
                    **request.json,
                    out_dir=project_cache_dir
                )
                wssw_process.start()
                assert wssw_process.is_alive()

                cache[project_id]['process'] = wssw_process
                cache[project_id]['participant'] = wss_worker
                
                # Created a resource        --> 201
                status = 201

            else:
                # Resource already exists   --> 200
                status = 200

            logging.info(f"Initialisation - Current state of Cache: {cache}")

            retrieved_metadata['is_live'] = cache[project_id]['process'].is_alive()
            
            expt_run_key = construct_combination_key(expt_id, run_id)

            if expt_run_key not in retrieved_metadata['connections']:
                retrieved_metadata['connections'].append(expt_run_key)
                
            if expt_run_key not in retrieved_metadata['in_progress']:
                retrieved_metadata['in_progress'].append(expt_run_key)

            updated_metadata = meta_records.update(
                project_id=project_id, 
                updates=retrieved_metadata
            )

            success_payload = payload_formatter.construct_success_payload(
                status=status,
                method="initialise.post",
                params=request.view_args,
                data=updated_metadata
            )
            return success_payload, status
    
        else:
            ns_api.abort(
                code=404, 
                message=f"Project logs '{project_id}' has not been initialised! Please poll and try again."
            )
