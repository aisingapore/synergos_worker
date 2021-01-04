#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
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
catalogue_template = cache_template['catalogue']

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
            1) Action (i.e. 'regress', 'classify', 'cluster', 'associate')
            2) Data tags

            eg. 

            {
                "action": 'classify'
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
                    'metadata': {},
                    'is_live': False,
                    'in_progress': [],
                    'connections': [],
                    'results': {}
                }
            )
            retrieved_metadata = meta_records.read(project_id)

        # If polling operation had already been done before, skip preprocessing
        # (Note: this is only valid if the submitted set of tags are the same)
        if retrieved_metadata['tags'] == request.json['tags']:
            data = retrieved_metadata   # retrieve aligned data from cache

        # Otherwise, perform preprocessing and archive results of operation
        else:
            # try:
            headers = {}
            schemas = {}
            metadata = {}
            exports = {}
            for meta, tags in request.json['tags'].items():

                if tags:

                    sub_keys = {'project_id': project_id, 'meta': meta}

                    # Prepare output directory for tensor export
                    project_meta_dir = outdir_template.safe_substitute(sub_keys)
                    project_cache_dir = os.path.join(project_meta_dir, "cache")
                    os.makedirs(project_cache_dir, exist_ok=True)

                    (
                        X_tensor, y_tensor, 
                        X_header, y_header, 
                        schema, 
                        df, 
                        meta_stats
                    ) = load_and_combine(
                        action=request.json['action'],
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
                    metadata[meta] = meta_stats

                    logging.debug(f"Exports: {exports}")

            # Export headers, schema & metadata extracted to "catalogue.json"
            catalogue_outpath = catalogue_template.safe_substitute({'project_id': project_id})            
            catalogue = {
                'headers': headers, 
                'schemas': schemas, 
                'metadata': metadata
            }
            logging.debug(f"-----> {catalogue}")
            with open(catalogue_outpath, 'w') as cp:
                json.dump(catalogue, cp)

            meta_records.update(
                project_id=project_id, 
                updates={
                    'tags': request.json['tags'],
                    'headers': headers,
                    'schemas': schemas,
                    'metadata': metadata, 
                    'exports': exports
                }
            )
            data = meta_records.read(project_id)
            logging.debug(f">>>> {data}")

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
