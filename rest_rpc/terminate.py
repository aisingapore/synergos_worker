#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import logging
import os
from pickle import GLOBAL
import signal
from pathlib import Path

# Libs
import jsonschema
import numpy as np
import psutil
import websockets
from flask import request, session
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.core.utils import Payload, MetaRecords, construct_combination_key
# from rest_rpc.initialise import cache, process, thread_condition, init_output_model
from rest_rpc.initialise import init_output_model

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

cache = app.config['CACHE']
thread_condition = app.config['THREAD_CONDITION']

# def on_terminate(proc):
#     print("process {} terminated with exit code {}".format(proc, proc.returncode))

# procs = psutil.Process().children()
# for p in procs:
#     p.terminate()
# gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)
# for p in alive:
    # p.kill()

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
        expt_run_key = construct_combination_key(expt_id, run_id)

        # Search local database for cached operations
        retrieved_metadata = meta_records.read(project_id)
                
        if (retrieved_metadata and 
            expt_run_key in retrieved_metadata['in_progress']):

            # Check that only 1 remaining experiment-runs is left in progress
            # (Note: Remaining experiment-run must be this experiment-run)
            if len(retrieved_metadata['in_progress']) == 1:
                
                process_id = int(os.getenv('process'))
                # global process
                # thread_condition.acquire()
                # logging.debug(f"Before deletion, cache: {cache.get_dict()}")
                # logging.debug(f"Before deletion, cache: {session}")
                logging.debug(f"Before deletion: process id - {process_id}")

                # if project_id in cache:
                # while True:
                    # process_id = cache.get(project_id)
                if process_id:

                    # process_id = project['process']
                    # process_id = app.config['PROCESS']

                    # project = session.pop(project_id)
                    # project = retrieved_metadata['']
                    # logging.debug(f"Project operations: {project}")
                    # process_id = retrieved_metadata['process']
                    wssw_process = psutil.Process(pid=process_id)
                    logging.debug(f"Before termination: Child processes: {wssw_process.children(recursive=True)}")

                    # wssw_process.send_signal(signal.SIGTERM)
                    # logging.debug(f"After termination: Child processes: {wssw_process.children(recursive=True)}")
                    # wss_worker = project['participant']

                    # wss_worker.remove_worker_from_local_worker_registry()

                    # # if wss_worker.loop.is_running():
                    #     wss_worker.loop.call_soon_threadsafe(
                    #         wss_worker.loop.stop
                    #     ).call_soon_threadsafe(
                    #         wss_worker.loop.close
                    #     )
                    #     assert not wss_worker.loop.is_running()
                        
                    logging.debug(f"wssw_process status: {wssw_process.status()}")
                    # if wssw_process.is_alive():
                    if wssw_process.is_running():
                        wssw_process.terminate()    # end the process
                        # wssw_process.join()         # reclaim resources from thread
                        exitcode = wssw_process.wait()
                        assert not wssw_process.is_running()

                        # logging.info(f"Terminated process status: {wssw_process.status()}")
                        logging.info(f"Terminated process id: {wssw_process.pid}")
                        logging.info(f"Terminated process exitcode: {exitcode}")
                        # logging.debug(f"After termination: Child processes: {wssw_process.children(recursive=True)}")
                        # assert not wssw_process.is_alive()
                        # wssw_process.close()  
                        # wssw_process.kill()

                    retrieved_metadata['process'] = None
                    retrieved_metadata['is_live'] = False
                    
                    # cache.delete(project_id)
                        # process = None

                        # thread_condition.notify_all()
                        # break

                    # else:
                    #     thread_condition.wait_for(lambda : cache.get(project_id))
                    #     # thread_condition.wait_for(lambda : app.config['PROCESS'] is not None)

                # thread_condition.release()

                del os.environ['process']

            # logging.info(f"Termination - Current state of Cache: {cache.get_dict()}")
            # logging.info(f"Termination - Current state of Cache: {session}")

            retrieved_metadata['in_progress'].remove(expt_run_key)

            logging.info(f"Termination - Final metadata state (retrieved): {retrieved_metadata}")

            updated_metadata = meta_records.update(
                project_id=project_id, 
                updates=retrieved_metadata
            )

            logging.info(f"Termination - Final metadata state (archived): {updated_metadata}")
            
            import time
            time.sleep(5)

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="terminate.post",
                params=request.view_args,
                data=updated_metadata
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Project '{project_id}' has not been initialised! Please poll and try again."
            )
