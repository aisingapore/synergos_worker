#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Flask, Blueprint
from flask_restx import Api

# Custom

##################
# Configurations #
##################

app = Flask(__name__)
app.config.from_object('config')

blueprint = Blueprint("PySyft WSSW Controller", __name__)
api_version = app.config['API_VERSION']
api = Api(
    app=blueprint,
    version=api_version,
    title="PySyft Worker REST-RPC Controller API", 
    description="Controller API to facilitate model training in a PySyft grid"
)

from rest_rpc.poll import ns_api as poll_ns
from rest_rpc.align import ns_api as align_ns
from rest_rpc.initialise import ns_api as initialise_ns
from rest_rpc.terminate import ns_api as terminate_ns
from rest_rpc.predict import ns_api as predict_ns

api.add_namespace(poll_ns, path="/poll")
api.add_namespace(align_ns, path="/align")
api.add_namespace(initialise_ns, path="/initialise")
api.add_namespace(terminate_ns, path="/terminate")
api.add_namespace(predict_ns, path="/predict")

app.register_blueprint(blueprint, url_prefix='/worker')
