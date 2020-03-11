#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import json
import os
import logging
import traceback
from collections import OrderedDict
from pathlib import Path

# Libs
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields, abort
from tinydb import TinyDB, Query

# Custom
from config import server_params
from initialise_server import load_and_combine, annotate, start_proc

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

app = Flask(__name__)
#api = Api(
#    app#, 
    #version="1.0",
    #title="PySyft WSSW Controller API", 
    #description="Controller API to facilitate model training in a PySyft grid"
#)
#ns = api.namespace("pysyft_wssw_ctrl", description='Something')

database = server_params['database']
cache = server_params['cache']

####################
# Helper Functions #
####################

def format_json(json):
    """ Takes in response json and parse into model-ingestable format
    Args:
        json (dict): Json retrieved from HTTP response 
    Returns:
        X_vals (np.ndarray)
        y_vals (np.ndarray)
    """
    json_ = {k:[v] for k,v in json.items()}

    return json_

##################
# Core Functions #
##################

# implement your routes flask app routes here
# Make sure you have both the following routes which should accept POST requests
# /train
# /predict

@app.route('/')
def index():
    return "This is a worker container!"


@app.route('/poll', methods=['GET'])
def poll():
    """ Retrieves specified metadata regarding the worker. Supported metadata
        include:
        1) Headers
        2) Schema
    """
    results = {'success': 0}

    # Obtain metadata-related queries for assembling MLFlow configuration file
    if request.json:
        """ JSON received will contain the following information:
            1) Data tags
            2) Data schema

            eg. 

            {
                "id": "Alice",
                "tags": {
                    "train": [["type_a","v2"], ["type_b","v3"]],
                    "evaluate": []
                }
            }
        """
        try:
            headers = {}
            schemas = {}
            for meta, tags in request.json['tags'].items():

                if tags:
                    _, _, X_header, y_header, schema = load_and_combine(
                        tags=tags
                    )
                    headers[meta] = {'X': X_header, 'y': y_header}
                    schemas[meta] = schema

            results['headers'] = headers
            results['schemas'] = schemas
            results['success'] = 1

            return jsonify(results), 200

        except KeyError:
            logging.error("Insufficient info specified for metadata tracing!")
            results['trace'] = traceback.format_exc()

            return jsonify(results), 400


@app.route('/register', methods=['POST'])
def register():
    """ Registers worker for a specified experiment
    """
    raise NotImplementedError(
        "Registration can only be done on TTP!"
    )


@app.route('/align/<expt_id>', methods=['POST'])
def align(expt_id):
    """ Receives and caches the null indexes to be populated within the worker's
        registered datasets in order for them to have the same schema as all 
        other participants
        Assumption: Worker's server parameters & schemas of registered datasets
                    have already be uploaded to TTP. This ensures that the TTP
                    has the necessary information to conduct multiple feature
                    alignment, as well as contact the respective workers
                    involved post-alignment.  
    """
    results = {'success': 0}

    # Handling TTP's header modification recommendations after MFA
    if request.json:
        """ JSON received will contain the following information:
            1) Run ID
            2) Training data tags
            3) Evaluation data tags
            3) Indexes to insert null representation of columns on

            eg. 

            {
                "run_id": "run_1",
                "tags": [["type_a","v2"]],
                "alignment": [0,1,3,6,8]
            }

            This data will be cached in the worker database for subsequent use.
        """
        try:
            run_id = request.json['run_id']

            with database as db:
                
                expt_table = db.table(expt_id)

                Experiment = Query()
                expt_table.upsert(request.json, Experiment.run_id == run_id)

                results['success'] = 1
                
                return jsonify(results), 201
        
        except KeyError:
            results['trace'] = traceback.format_exc()
            
            return jsonify(results), 400


@app.route('/initialise/<expt_id>/<run_id>', methods=['POST'])
def initialise(expt_id, run_id):
    """ Start up WebsocketServerWorker for representing this worker container
    
    Args:
        expt_id (str): Identifier of experiment
        run_id (str): Identifier of run within experiment
    """ 
    results = {'success': 0}

    if request.json:
        """ JSON received will contain the following information:
            1) Run ID
            2) Worker ID
            3) Host IP
            4) Host Port
            5) Verbosity

            eg.

            {
                "host": "0.0.0.0"
                "port": 8020,
                "id": "worker_0",
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
                },
                "log_msgs": true,
                "verbose": true
            }
        """
        try:
            # Check that specified experiment run is not already running
            if not cache[expt_id][run_id]:

                """
                IMPLEMENT JSON SCHEMAS TO CHECK FOR CONSISTENT STRUCTURE!!!
                """
                wssw_process, wss_worker = start_proc(**request.json)
                wssw_process.start()
                assert wssw_process.is_alive()

                cache[expt_id][run_id]['process'] = wssw_process
                cache[expt_id][run_id]['participant'] = wss_worker
                results['success'] = 1

                return jsonify(results), 200

        except Exception:
            results['trace'] = traceback.format_exc()
            
            return jsonify(results), 400


@app.route('/terminate/<expt_id>/<run_id>', methods=['POST'])
def terminate(expt_id, run_id):
    """ Closes WebsocketServerWorker to prevent potential cyber attacks during
        times of inactivity

    Args:
        expt_id (str): Identifier of experiment
        run_id (str): Identifier of run within experiment
    """
    results = {'success': 0}

    try:
        expt_run = cache[expt_id].pop(run_id)

        wssw_process = expt_run['process']
        wss_worker = expt_run['participant']

        if wss_worker.loop.is_running():
            wss_worker.loop.stop().close()

        if wssw_process.is_alive():
            wssw_process.terminate()

        results['success'] = 1

        return jsonify(results), 200

    except KeyError:
        logging.error(f"Experiment {expt_id} run {run_id} is not running. Unable to terminate!")
        results['trace'] = traceback.format_exc()
        
        return jsonify(results), 400

# CHANGE ERROR CODE FOR EXCEPTIONS!
# Add in JSON Schemas

@app.route('/predict/<expt_id>/<run_id>', methods=['POST'])
def predict(expt_id, run_id):
    """
    """
    results = {'success': 0}
    try:
        results['success'] = 1
        
        return jsonify(results), 200

    except Exception as e:
        print('Predict: Error occurred!', e)
        results['trace'] = traceback.format_exc()

        return jsonify(results), 400


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0")


##############
# Deprecated #
##############
"""
@app.route('/align', methods=['GET','POST'])
def align():
    # Handling TTP's header retrieval request for Multiple Feature Alignment
    if request.method == 'GET':
        return jsonify({'trace': traceback.format_exc()})
    else:
        pass



    parser = argparse.ArgumentParser(
        description="REST orchestrator for Worker Node."
    )

    parser.add_argument(
        "--id",
        "-i",
        type=str,
        required=True,
        help="Name (id) of the websocket server worker, e.g. --id Alice"
    )

    kwargs = vars(parser.parse_args())
    logging.info(f"Worker Parameters: {kwargs}")

    with database as db:

        cache_table = db.table('Cache')
        cache_table.insert(kwargs)
"""