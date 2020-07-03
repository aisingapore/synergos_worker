#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import asyncio
import json
import logging
import os
from glob import glob
from multiprocessing import Event, Process
from pathlib import Path

# Libs
import pandas as pd
import syft as sy
import torch as th
from syft.workers.websocket_server import WebsocketServerWorker

# Custom
from rest_rpc import app
from rest_rpc.core.pipelines import (
    Preprocessor, 
    TabularPipe, 
    ImagePipe, 
    TextPipe
)

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

hook = sy.TorchHook(th, verbose=False) # toggle where necessary

data_dir = app.config['DATA_DIR']
out_dir = app.config['OUT_DIR']

#############
# Functions #
#############

def detect_metadata(tag):
    """ Retrieves participant defined metadata describing their declared 
        datasets, as well as possible special operations they would like to
        include or skip during preprocessing. All options and descriptors are
        specified in a file called `metadata.json`, which MUST reside in the
        same folder as the dataset itself.

    Args:
        tag (list(str)): Tag of dataset to load into worker
    Returns:
        All metadata declared along a specific tag
    """
    # data_dir = "/home/chompsterz/Desktop/datasets"
    core_dir = Path(data_dir)
    all_metadata_paths = list(core_dir.glob("**/metadata.json"))

    all_metadata = {}
    for meta_path in all_metadata_paths:

        if set(tag).issubset(set(meta_path.parts)):

            with open(meta_path, 'r') as s:
                metadata = json.load(s)

            all_metadata[meta_path.parent] = metadata

    return all_metadata


def load_tabulars(tab_dir: str, metadata: dict, out_dir: str) -> pd.DataFrame:
    """ Loads in all tabular datasets found in the specified tagged directory.
        Tabular datasets are expected to exist as `.csv` files. While it is
        preferable to separate and specify different `.csv` files under
        different token tags, the system can handle multiple `.csv` files in a
        single directory, on the condition that they MUST have the same:
        
        1) Schema       (i.e. no. of features)
        2) Datatypes    (i.e. type encoded for each feature value)
        3) Preprocessing priorities and exclusions

        For example, participants are expected to organise their tabular 
        datasets like this:
        
        eg.
            /path
                /matching
                    /declared
                        /tag_recommended
                            dataset_1
                            schema.json
                            metadata.json
                        /tag_unrecommended_but_supported
                            dataset_2_1.csv
                            dataset_2_2.csv
                            schema.json
                            metadata.json

        IMPORTANT: 
        `metadata.json` and `schema.json` are both expected to reside in the
        same folder
    """
    core_dir = Path(tab_dir)
    all_tab_paths = list(core_dir.glob("**/*.csv"))

    datasets = []
    relevant_data_paths = []
    for _path in all_tab_paths:

        data = pd.read_csv(_path)

        # Retrieve corresponding schema path. Schemas as supposed to be
        # stored in the same directory as the dataset(s). At each root
        # classification, there can be more than 1 dataset, but all datasets
        # must have the same schema.
        schema_path = os.path.join(_path.parent, "schema.json")
        with open(schema_path, 'r') as s:
            schema = json.load(s)

        ##################################################
        # Edge Case: No seeding values for interpolation #
        ##################################################

        # [Cause]
        # This happens when there are feature columns within the original 
        # dataset that have no values at all (i.e. all NAN values). Being a 
        # NAN slice, CatBoost will not have any values to infer trends on, 
        # and will not be able to interpolate those aflicted features. 
        # CatBoost WILL NOT raise errors, since it deems NANs as valid 
        # types. As a result, even after inserting spacer columns (obtained 
        # from performing multiple feature alignment) on one-hot-encoded 
        # matrices, there will still be NAN values present.

        # [Problems]
        # Feeding NAN values into models & criterions for loss calculation 
        # will cause errors, breaking the FL training cycle and halting the 
        # grid.

        # [Solution]
        # Remove features corresponding to NAN slices entirely from dataset.
        # This allows a true representation of the participant's data to be
        # propagated down the pipeline, which can be caught & penalised
        # accordingly by the contribution calculator downstream.

        # Augment schema to cater to condensed dataset
        na_slices = data.columns[data.isna().all()].to_list()
        logging.debug(f"NA slices: {na_slices}")

        condensed_schema = {
            feature: d_type for feature, d_type in schema.items()
            if feature not in na_slices
        }
        condensed_data = data.dropna(axis='columns', how='all')
        assert set(condensed_schema.keys()) == set(condensed_data.columns)
        logging.debug(f"Condensed columns: {list(condensed_data.columns)}, length: {len(condensed_data.columns)}")

        preprocessor = Preprocessor(
            data=condensed_data, 
            schema=condensed_schema, 
            train_dir=out_dir
        )
        interpolated_data = preprocessor.interpolate()

        datasets.append(interpolated_data)
        relevant_data_paths.append(_path)

    logging.debug(f"Data paths detected: {relevant_data_paths}")

    if datasets:
        tag_unified_data = pd.concat(
            datasets, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        logging.debug(f"Tag-unified Schema: {tag_unified_data.dtypes.to_dict()}")

        return tag_unified_data

    else:
        raise RuntimeError("No valid datasets were detected!")


def load_images(img_dir: str, metadata: dict) -> pd.DataFrame:
    """ Loads in all image datasets found in the specified tagged directory.
        Image datasets are expected to be organised according to their classes.
        For example, if a dataset consists of 3 classes, the folder structure
        that participants are expected to organise their images like this:
        
        eg.
            /path
                /matching
                    /declared
                        /tag
                            /class_1
                                image_1.png
                                image_2.png
                            /class_2
                                image_3.png
                            /class_3
                                image_4.png
                            metadata.json
                        /another_tag
                            /class_1
                                image_5.jpeg
                        /one_last_tag
                            /class_4
                                image_6.gif

        IMPORTANT: `metadata.json` is expected to reside in the same folder.
    """
    class_images = {}
    for element in os.listdir(img_dir):

        class_dir = Path(os.path.join(img_dir, element)).resolve()

        # Based on the dictated file structure, images are sorted in directories
        # corresponding to their classes. Hence, extract these directories as
        # class names. Ignore `metadata.json`.
        if os.path.isdir(class_dir):
            image_paths = [x for x in class_dir.glob("**/*") if x.is_file()]
            class_images[element] = image_paths

    if class_images:
        img_pipeline = ImagePipe(data=class_images)
        img_pipeline.run()

        return img_pipeline.output

    else:
        raise RuntimeError("No valid image datasets were detected!")


def load_texts(txt_dir: str, metadata: dict) -> pd.DataFrame:
    """ Loads in all corpora found in the specified tagged directory.
        Corpora are expected to exist as `.csv` files with only 2 columns, the
        `text` and `target`. Multiple corpora are allowed in the same directory.

        For example, participants are expected to organise their corpora like 
        this:
        
        eg.
            /path
                /matching
                    /declared
                        /tag
                            corpus_1.csv
                            corpus_2.csv
                            corpus_3.csv
                            metadata.json

        IMPORTANT: 
        `metadata.json` is expected to reside in the direct child folder of the 
        declared tagged directory.
    """
    core_dir = Path(txt_dir)
    all_txt_paths = list(core_dir.glob("**/*.csv"))

    if all_txt_paths:
        text_pipeline = TextPipe(data=all_txt_paths)
        text_pipeline.run()

        return text_pipeline.output

    else:
        raise RuntimeError("No valid corpora were detected!")


def load_dataset(tag, out_dir=out_dir):
    """ Loads in all datasets found in the specified tagged directory.
        Note: A tag is defined as a list of n tokens, each token corresponding
            to a sub-classification of datasets
            eg. ["type_A", "v1"] corresponds to "~/data/type_A/v1/data.csv"
                ["type_B"] corresponds to "~/data/type_B/data.csv"

        IMPORTANT: All datasets detected along the declared tag MUST be of the
                   SAME datatype!
    Args:
        tag (list(str)): Tag of dataset to load into worker
    Returns:
        Tag-unified dataset (pd.DataFrame)
    """
    all_metadata = detect_metadata(tag)

    if all_metadata:

        # Ensure that all datasets detected under the tags are of the same datatype
        # This assumption might be revised later
        assert len(set([m['datatype'] for _,m in all_metadata.items()])) == 1

        datasets = []
        for core_dir, metadata in all_metadata.items():

            datatype = metadata['datatype']

            if datatype == "tabular":
                df = load_tabulars(core_dir, metadata, out_dir)
            elif datatype == "image":
                df = load_images(core_dir, metadata)
            elif datatype == "text":
                df = load_texts(core_dir, metadata)
            else:
                raise ValueError(f"Specified Datatype {datatype} is not supported!")

            datasets.append(df)

        tag_unified_data = pd.concat(
            datasets, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        return tag_unified_data

    else:
        raise RuntimeError("No valid datasets were detected!")


def load_and_combine(tags, out_dir=out_dir):
    """ Loads in all datasets found along the corresponding subdirectory
        sequence defined by the specified tags, and combines them into a single
        unified dataset.
        Assumption: 1) All datasets form the same tag sequence must have the 
                       same shape (i.e. completely-feature aligned)
                    2) Same features across datasets should have the same 
                       declared datatypes
    
    Args:
        tag  (list(list(str))): Tags of datasets to load into worker
    Returns:
        X_combined_tensor (th.Tensor)
        y_combined_tensor (th.Tensor)
        X_header (list(str))
        y_header (list(str))
        Combined Schema (dict(str))
        combined_df (pd.DataFrame)
    """
    tag_unified_datasets = [load_dataset(tag=tag, out_dir=out_dir) 
                            for tag in tags]

    combined_schema = {}
    for df in tag_unified_datasets:
        logging.debug(f"Loaded dataset size: {len(df.columns)}")
        curr_schema = df.dtypes.apply(lambda x: x.name).to_dict()
        combined_schema.update(curr_schema)

    combined_data = pd.concat(
        tag_unified_datasets, 
        axis=0
    ).drop_duplicates().reset_index(drop=True)
    logging.debug(f"Combined data size: {len(combined_data.columns)}")

    preprocessor = Preprocessor(
        combined_data, 
        combined_schema, 
        train_dir=out_dir
    )
    preprocessor.interpolate()
    logging.debug(f"Interpolated combined data: {preprocessor.output}")
    X, y, X_header, y_header = preprocessor.transform()

    X_combined_tensor = th.Tensor(X)
    y_combined_tensor = th.Tensor(y)

    return (
        X_combined_tensor, 
        y_combined_tensor, 
        X_header, 
        y_header, 
        combined_schema,
        preprocessor.output
    )


def annotate(X, y, id, meta):
    """ Annotate a specified data tensor with its corresponding metadata 
        required for federated training.

    Args:
        X (th.Tensor): Feature tensor to be annotated
        y (th.Tensor): Target tensor to be annotated
        id (str): Identifier of participant's WebsocketServerWorker 
        meta (str): Classification of dataset (i.e. train/evaluate)
    Returns:
        Annotated dataset (th.Tensor)
    """
    X_annotated = X.tag(
        "#X",
        f"#{meta}"
    ).describe(f"{meta}: Predictor values contributed by {id}")

    y_annotated = y.tag(
        "#y", 
        f"#{meta}"
    ).describe(f"{meta}: Target values contributed by {id}")

    return X_annotated, y_annotated


def start_proc(participant=WebsocketServerWorker, out_dir=out_dir, **kwargs):
    """ helper function for spinning up a websocket participant 
    
    Args:
        participant (WebsocketServerWorker): Type of worker to instantiate
        **kwargs: Parameters to be used for instantiation
    Returns:
        Federated worker
    """

    def align_dataset(dataset, alignment_idxs):
        """ Takes in a dataset & inserts null columns in accordance to MFA
            defined spacer indexes. Alignment indexes are sorted in ascending
            order.

        Args:
            dataset (th.Tensor): Data tensor to be augmented
            alignment_idxs (list(int)): Spacer indexes where dataset should
                                        insert null columns in order to properly
                                        align dataset to model trained
        Returns:
            Augmented dataset (th.Tensor)
        """
        augmented_dataset = dataset.clone()
        for idx in alignment_idxs:

            logging.debug(f"Current spacer index: {idx}")
            logging.debug(f"Before augmentation: size is {augmented_dataset.shape}")

            # Slice the dataset along the specified inclusion index
            first_segment = augmented_dataset[:, :idx]
            second_segment = augmented_dataset[:, idx:]
            logging.debug(f"First segment's shape: {first_segment.shape}")
            logging.debug(f"Second segment's shape: {second_segment.shape}")

            # Generate spacer column
            new_col = th.zeros(dataset.shape[0], 1)
            logging.debug(f"Spacer column's shape: {new_col}, {new_col.shape}")

            # Concatenate all segments together, using the spacer as partition
            augmented_dataset = th.cat(
                (first_segment, new_col, second_segment),
                dim=1
            )

            logging.debug(f"After augmentation: size is {augmented_dataset.shape}")
        return augmented_dataset

    def target(server):
        """ Initialises websocket server to listen to specified port for
            incoming connections

        Args:
            server (WebsocketServerWorker): WS worker representing participant
        """
        server.start()
        
    all_tags = kwargs.pop('tags')
    all_alignments = kwargs.pop('alignments')

    final_datasets = tuple()
    for meta, tags in all_tags.items():

        X, y, _, _, _, _ = load_and_combine(tags=tags, out_dir=out_dir)
        logging.debug(f"Start process - X shape: {X.shape}")

        feature_alignment = all_alignments[meta]['X']
        logging.debug(f"Start process - feature alignment indexes: {feature_alignment}")
        X_aligned = align_dataset(X, feature_alignment)

        target_alignment = all_alignments[meta]['y']
        y_aligned = align_dataset(y, target_alignment)

        X_aligned_annotated, y_aligned_annotated = annotate(
            X=X_aligned, 
            y=y_aligned, 
            id=kwargs['id'], 
            meta=f"{meta}"
        )
        final_datasets += (X_aligned_annotated, y_aligned_annotated)

        logging.debug(f"Loaded {meta} data: {X_aligned_annotated, y_aligned_annotated}")

    kwargs['data'] = final_datasets  #[X, y]
    logging.info(f"Worker metadata: {kwargs}")

    logging.debug(f"Before participant initialisation - Registered workers in grid: {hook.local_worker._known_workers}")
    logging.debug(f"Before participant initialisation - Registered workers in env : {sy.local_worker._known_workers}")

    # Originally, the WSS worker could function properly without mentioning
    # a specific event loop. However, that's because it was ran as the main
    # process, on the default event loop. However, since WSS worker is now 
    # initialised as a child process, there is a need to define its very own
    # event loop to separately drive the WSSW process.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # The initialised loop is then fed into WSSW as a custom governing loop
    # Note: `auto_add=False` is necessary here because we want the WSSW object
    #       to get automatically garbage collected once it is no longer used.
    kwargs.update({'loop': loop})
    server = participant(hook=hook, **kwargs)
    # server.broadcast_queue = asyncio.Queue(loop=loop)

    p = Process(target=target, args=(server,))

    # Ensures that when process exits, it attempts to terminate all of its 
    # daemonic child processes.
    p.daemon = True

    logging.debug(f"After participant initialisation - Registered workers in grid: {hook.local_worker._known_workers}")
    logging.debug(f"After participant initialisation - Registered workers in env : {sy.local_worker._known_workers}")

    return p, server

##########
# Script #
##########

if __name__ == "__main__":
    
    # tags = [["edge_test_missing_coecerable_vals"], ["edge_test_misalign"]]
    # print(load_and_combine(tags))

    tabular_tags = [["tabular", "heartdisease_federated", "data1"]]#, "edge_test_missing_coecerable_vals"]]
    # for tag in tabular_tags:
    #     all_metadata = detect_metadata(tag)
    #     print(all_metadata)

    #     for _dir, metadata in all_metadata.items():
    #         print(load_tabulars(_dir, metadata))

    image_tags = [
        ["image", "mnist_federated", "data1", "train"],
        ["image", "mnist_federated", "data1", "predict"]
    ]
    # for tag in image_tags:
    #     all_metadata = detect_metadata(tag)

    #     for _dir, metadata in all_metadata.items():
    #         print(load_images(_dir, metadata))
    
    # print(load_dataset(tabular_tags[0]))
    print(load_and_combine(image_tags))
    """
    parser = argparse.ArgumentParser(description="Run websocket server worker.")

    parser.add_argument(
        "--host",
        "-H",
        type=str, 
        default="localhost", #'172.16.2.11', 
        help="Host for the connection"
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int, 
        default=8020,
        help="Port number of the websocket server worker, e.g. --port 8020"
    )

    parser.add_argument(
        "--id",
        "-i",
        type=str,
        required=True,
        help="Name (id) of the websocket server worker, e.g. --id Alice"
    )

    parser.add_argument(
        "--train",
        "-t",
        type=str,
        nargs="+",
        required=True,
        help="Dataset Tag(s) to load into worker for TRAINING"
    )

    parser.add_argument(
        "--evaluate",
        "-e",
        nargs="+",
        type=str,
        help="Dataset Tag(s) to load into worker for Evaluation"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode"
    )

    kwargs = vars(parser.parse_args())
    logging.info(f"Worker Parameters: {kwargs}")
    
    if os.name != "nt":
        server = start_proc(WebsocketServerWorker, kwargs)
    else:
        server = WebsocketServerWorker(**kwargs)
        server.start()
    """