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
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.core.utils import Payload, MetaRecords
from rest_rpc.initialise import cache, init_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "terminate", 
    description='API to faciliate WSSW termination for participant.'
)

db_path = app.config['DB_PATH']
meta_records = MetaRecords(db_path=db_path)

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
        """
        expt_run_key = str((expt_id, run_id))

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

                if wss_worker.loop.is_running():
                    wss_worker.loop.call_soon_threadsafe(
                        wss_worker.loop.stop
                    ).call_soon_threadsafe(
                        wss_worker.loop.close
                    )

                if wssw_process.is_alive():
                    #wssw_process.terminate()    # end the process
                    wssw_process.kill()
                    logging.info(f"Terminated process id: {wssw_process.pid}")
                    logging.info(f"Terminated process exitcode: {wssw_process.exitcode}")
                    wssw_process.join()         # reclaim resources from thread

                assert not wssw_process.is_alive()
                assert not wss_worker.loop.is_running()
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
                params={
                    'project_id': project_id,
                    'expt_id': expt_id,
                    'run_id': run_id
                },
                data=updated_metadata
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' has not been initialised! Please poll and try again."
            )
