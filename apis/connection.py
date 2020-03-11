#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
from flask import jsonify
from flask_restx import Namespace, Resource, fields, reqparse

# Custom
from config import server_params
from initialise_server import load_and_combine, annotate, start_proc

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns = Namespace(
    "preprocessing", 
    description='API to faciliate grid configuration & connection.'
)

parser = reqparse.RequestParser()
parser.add_argument('rate', type=int, help='Rate to charge for this resource')
args = parser.parse_args()

##########
# Models #
##########

# Models are used for marshalling (i.e. moulding responses)

tags = ns.model(
    name="tags",
    model={
        'train': fields.List(fields.List(fields.String), required=True),
        'evaluate': fields.List(fields.List(fields.String), required=True),
        'predict': fields.List(fields.List(fields.String), required=True)
    }
)

#############
# Resources #
#############

@ns.route('/')
class Index(Resource):
    """
    """
    @ns.doc("get_index")
    def get(self):
        return "This is a worker container!"

@ns.route('/poll')
@ns.param('', "")
@ns.response()
class Poll(Resource):
    """
    Retrieves specified metadata regarding the worker. 
    Supported metadata include:
        1) Headers
        2) Schema
    """

    @ns.doc("")
    @ns.param("", description="")
    @ns.marshal_with(tags, code=200)
    def get(self):
        results = {'success': 0}
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

        except KeyError:
            logging.error("Insufficient info specified for metadata tracing!")
    
        return jsonify(results)