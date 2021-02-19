#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import logging
import os
import signal
from pathlib import Path

# Libs
import jsonschema
import numpy as np
import websockets
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.core.utils import Payload, MetaRecords, construct_combination_key
from rest_rpc.initialise import cache, init_output_model

# Synergos logging
from SynergosLogger.init_logging import logging

##################
# Configurations #
##################


ns_api = Namespace(
    "terminate", 
    description='API to faciliate WSSW termination for participant.'
)

db_path = app.config['DB_PATH']
meta_records = MetaRecords(db_path=db_path)

logging.info(f"terminate.py logged", Description="Changes made")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Imported from initialise.py

payload_formatter = Payload('Terminate', ns_api, init_output_model)

#############
# Resources #
#############

@ns_api.route('/<project_id>/<expt_id>/<run_id>')
@ns_api.response(200, "WSSW object successfully terminated")
@ns_api.response(404, "WSSW object not found")
@ns_api.response(500, 'Internal failure')
class Termination(Resource):

    @ns_api.doc("terminate_wssw")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def post(self, project_id, expt_id, run_id):
        """ Closes WebsocketServerWorker to prevent potential cyber attacks during
            times of inactivity

            JSON received will contain the following information:
            1) Connections

            eg.

            {
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
                }
            }
        """
        expt_run_key = construct_combination_key(expt_id, run_id)

        # Search local database for cached operations
        retrieved_metadata = meta_records.read(project_id)
                
        if (retrieved_metadata and 
            expt_run_key in retrieved_metadata['in_progress']):

            # Check that only 1 remaining experiment-runs is left in progress
            # (Note: Remaining experiment-run must be this experiment-run)
            if len(retrieved_metadata['in_progress']) == 1:

                project = cache.pop(project_id)
                wssw_process = project['process']
                wss_worker = project['participant']

                wss_worker.remove_worker_from_local_worker_registry()

                if wss_worker.loop.is_running():
                    wss_worker.loop.call_soon_threadsafe(
                        wss_worker.loop.stop
                    ).call_soon_threadsafe(
                        wss_worker.loop.close
                    )
                    assert not wss_worker.loop.is_running()
                    
                if wssw_process.is_alive():
                    wssw_process.terminate()    # end the process
                    wssw_process.join()         # reclaim resources from thread
                    logging.info(f"Terminated process id: {wssw_process.pid}", Class=Termination.__name__, function=Termination.post.__name__)
                    logging.info(f"Terminated process exitcode: {wssw_process.exitcode}", Class=Termination.__name__, function=Termination.post.__name__)
                    assert not wssw_process.is_alive()
                    wssw_process.close()  

                retrieved_metadata['is_live'] = False

            logging.info(f"Termination - Current state of Cache: {cache}")

            retrieved_metadata['in_progress'].remove(expt_run_key)

            updated_metadata = meta_records.update(
                project_id=project_id, 
                updates=retrieved_metadata
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="terminate.post",
                params=request.view_args,
                data=updated_metadata
            )
            logging.info(f"Successful payload", code="200", Class=Termination.__name__, function=Termination.post.__name__)
            return success_payload, 200

        else:
            logging.error(f"Project not initialised", code="404", description=f"Project logs '{project_id}' has not been initialised! Please poll and try again.", Class=Termination.__name__, function=Termination.post.__name__)

            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' has not been initialised! Please poll and try again."
            )
