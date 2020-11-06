#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from pathlib import Path

# Libs


# Custom
from rest_rpc import app
from rest_rpc.core.utils import Records

##################
# Configurations #
##################

tag_details = {
    "train": [["type_a","v1"], ["type_b","v2"]],
    "evaluate": [["type_c","v3"]]
}

header_details = {
    "train": {
        "X": ["X1_1", "X1_2", "X2_1", "X2_2", "X3"],
        "y": ["target_1", "target_3"]
    },
    "evaluate": {
        "X": ["X1_1", "X1_2", "X2_1", "X3"],
        "y": ["target_1", "target_2"]
    }
}

schema_details = {
    "train": {
        "X1": "category",
        "X2": "category", 
        "X3": "int32", 
        "target": "category"
    },
    "evaluate": {
        "X1": "category",
        "X2": "category", 
        "X3": "int32", 
        "target": "category"
    }
}

export_details = {
    "train": {
        "X": "path/to/training/data/preprocessed_X.txt",
        "y": "path/to/training/data/preprocessed_y.txt"
    },
    "evaluate": {
        "X": "path/to/evaluation/data/preprocessed_X.txt",
        "y": "path/to/evaluation/data/preprocessed_y.txt"
    }
}

alignment_details = {
    "train": {
        "X": [0,1,3,6,8],
        "y": [1]
    },
    "evaluate": {
        "X": [0,1,3,6,8,9],
        "y": [2],
    }
}

poll_details = {
    "tags": tag_details,
    "headers": header_details,
    "schemas": schema_details,
    "exports": export_details
}

meta_updates_1 = {
    "alignments": alignment_details
}

meta_updates_2 = {
    "is_live": True,
    "connections": ["(expt_1, run_1)"],
    "in_progress": ["(expt_1, run_1)"]
}

project_id = "poly_collab"
expt_id = "3_layer_nn"
run_id = "run_002"

test_db_path = os.path.join(
    app.config['TEST_DIR'], 
    "archives", 
    "test_database.json"
)

#########################
#  Evaluation Functions #
#########################

def check_equivalence_and_format(record):
    assert 'created_at' in record.keys()
    record.pop('created_at')
    assert "key" in record.keys()
    key = record.pop('key')
    assert set([project_id]) == set(key.values())
    return record 


def check_detail_equivalence(details, template):
    assert details == template
    return details

#######################
# Records Class Tests #
#######################

def test_Records_create():
    records = Records(db_path=test_db_path)
    new_metadata = {'key': {"project_id": project_id}}
    new_metadata.update(poll_details)
    created_metadata = records.create(
        subject='Metadata',
        key='key',
        new_record=new_metadata
    )
    raw_details = check_equivalence_and_format(created_metadata)
    check_detail_equivalence(raw_details, poll_details)


def test_Records_read_all():
    records = Records(db_path=test_db_path)
    all_metadata = records.read_all(subject='Metadata')
    assert len(all_metadata) == 1
    retrieved_metadata = all_metadata[0]
    raw_details = check_equivalence_and_format(retrieved_metadata)
    check_detail_equivalence(raw_details, poll_details)


def test_Records_read():
    records = Records(db_path=test_db_path)
    retrieved_metadata = records.read(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    )
    assert retrieved_metadata is not None
    raw_details = check_equivalence_and_format(retrieved_metadata)
    check_detail_equivalence(raw_details, poll_details)


def test_MetaRecords_update():
    records = Records(db_path=test_db_path)
    targeted_metadata_1 = records.read(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    )
    updated_metadata_1 = records.update(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id},
        updates=meta_updates_1
    )
    assert targeted_metadata_1.doc_id == updated_metadata_1.doc_id
    for k,v in meta_updates_1.items():
        assert updated_metadata_1[k] == v
    targeted_metadata_2 = records.read(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    )
    updated_metadata_2 = records.update(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id},
        updates=meta_updates_2
    )
    assert targeted_metadata_2.doc_id == updated_metadata_2.doc_id
    for k,v in meta_updates_2.items():
        assert updated_metadata_2[k] == v


def test_Records_delete():
    records = Records(db_path=test_db_path)
    targeted_metadata = records.read(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    )
    deleted_metadata = records.delete(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    )
    assert targeted_metadata.doc_id == deleted_metadata.doc_id
    assert records.read(
        subject='Metadata',
        key='key',
        r_id={"project_id": project_id}
    ) is None
