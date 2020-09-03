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
from rest_rpc.core.server import load_and_combine
from rest_rpc.core.utils import Payload, MetaRecords

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "poll", 
    description='API to faciliate metadata retrieval from participant.'
)

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
meta_records = MetaRecords(db_path=db_path)

cache_template = app.config['CACHE_TEMPLATE']
outdir_template = cache_template['out_dir']
X_template = cache_template['X']
y_template = cache_template['y']
df_template = cache_template['dataframe']

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
        "tags": fields.Nested(tag_model, required=True)
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
    model={"*": schema_field}
)

schema_model = ns_api.model(
    name="schema",
    model={
        'train': fields.Nested(meta_model, required=True),
        'evaluate': fields.Nested(meta_model, skip_none=True),
        'predict': fields.Nested(meta_model, skip_none=True)
    }
)

poll_model = ns_api.model(
    name="poll",
    model={
        'headers': fields.Nested(header_model, required=True),
        'schemas': fields.Nested(schema_model, required=True)
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

@ns_api.route('/<project_id>')
@ns_api.response(200, "Initialised project logs successfully")
@ns_api.response(417, "Insufficient info specified for metadata tracing")
@ns_api.response(500, "Internal failure")
class Poll(Resource):
    """ Provides necessary information from participant for orchestration """

    @ns_api.doc("poll_metadata")
    @ns_api.expect(poll_input_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def post(self, project_id):
        """ Retrieves specified metadata regarding the worker.

            JSON received will contain the following information:
            1) Data tags

            eg. 

            {
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
                }
            }
        """
        # Search local database for cached operations
        retrieved_metadata = meta_records.read(project_id)

        # If polling operation had already been done before, skip preprocessing
        # (Note: this is only valid if the submitted set of tags are the same)
        if (retrieved_metadata and 
            retrieved_metadata['tags'] == request.json['tags']):
            data = retrieved_metadata

            # Retrieve aligned data from cache

        # Otherwise, perform preprocessing and archive results of operation
        else:
            # try:
            headers = {}
            schemas = {}
            exports = {}
            for meta, tags in request.json['tags'].items():

                if tags:

                    sub_keys = {'project_id': project_id, 'meta': meta}

                    # Prepare output directory for tensor export
                    project_meta_dir = outdir_template.safe_substitute(sub_keys)
                    project_cache_dir = os.path.join(project_meta_dir, "cache")
                    os.makedirs(project_cache_dir, exist_ok=True)

                    (X_tensor, y_tensor, X_header, y_header, schema, df
                    ) = load_and_combine(
                        tags=tags, 
                        out_dir=project_cache_dir,
                        is_condensed=False
                    )

                    logging.debug(f"Polled X_header: {X_header}")
                    logging.debug(f"Polled y_header: {y_header}")

                    # Export X & y tensors for subsequent use
                    X_export_path = X_template.safe_substitute(sub_keys)
                    with open(X_export_path, 'wb') as xep:
                        np.save(xep, X_tensor.numpy())

                    y_export_path = y_template.safe_substitute(sub_keys)
                    with open(y_export_path, 'wb') as yep:
                        np.save(yep, y_tensor.numpy())

                    # Export combined dataframe for subsequent use
                    df_export_path = df_template.safe_substitute(sub_keys)
                    df.to_csv(df_export_path, encoding='utf-8')

                    exports[meta] = {
                        'X': X_export_path, 
                        'y': y_export_path,
                        'dataframe': df_export_path
                    }
                    headers[meta] = {'X': X_header, 'y': y_header}
                    schemas[meta] = schema

                    logging.debug(f"Exports: {exports}")


            data = meta_records.create(
                project_id=project_id, 
                details={
                    'tags': request.json['tags'],
                    'headers': headers,
                    'schemas': schemas,
                    'exports': exports,
                    'is_live': False,
                    'in_progress': [],
                    'connections': [],
                    'results': {}
                }
            )

            # except KeyError:
            #     ns_api.abort(                
            #         code=417,
            #         message="Insufficient info specified for metadata tracing!"
            #     )

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="poll.post",
            params=request.view_args,
            data=data
        )
        return success_payload, 200
