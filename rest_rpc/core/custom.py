#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import binascii
import json
import os
from glob import glob
from multiprocessing import Event, Process
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Libs
import pandas as pd
import syft as sy
import torch as th
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.object_storage import ObjectStore
from syft.workers.abstract import AbstractWorker
from syft.workers.message_handler import BaseMessageHandler
from syft.workers.websocket_server import WebsocketServerWorker

# Custom
from rest_rpc import app

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

hook = sy.TorchHook(th, is_client=False)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("custom.py logged", Description="No Changes")

###########################################
# Custom Helper Class - CustomObjectStore #
###########################################

class CustomObjectStore(ObjectStore):

    def __init__(self, owner: AbstractWorker):
        super().__init__(owner=owner)

    
    def get_obj(self, obj_id: Union[str, int]) -> object:
        """Returns the object from registry.
        Look up an object from the registry using its ID.
        Args:
            obj_id: A string or integer id of an object to look up.
        Returns:
            Object with id equals to `obj_id`.
        """

        try:
            obj = self._objects[obj_id]
        except KeyError as e:
            # if obj_id not in self._objects:
            #     raise ObjectNotFoundError(obj_id, self)
            # else:
            #     raise e

            logging.warning(f"object {obj_id} not found! Error: {e}")
            obj_id = None

        return obj



###########################################
# Custom Async Class - CustomServerWorker #
###########################################

class CustomServerWorker(WebsocketServerWorker):
    """ This is a simple extension to WebsocketServerWorkers in allow
        asynchronous fitting with generic algorithm suppport

    Args:
        hook (sy.TorchHook): a normal TorchHook object
        id (str or id): the unique id of the worker (string or int)
        log_msgs (bool): whether or not all messages should be
            saved locally for later inspection.
        verbose (bool): a verbose option - will print all messages
            sent/received to stdout
        host (str): the host on which the server should be run
        port (int): the port on which the server should be run
        data (dict): any initial tensors the server should be
            initialized with (such as datasets)
        loop: the asyncio event loop if you want to pass one in
            yourself
        cert_path: path to used secure certificate, only needed for secure connections
        key_path: path to secure key, only needed for secure connections
    """
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[th.Tensor, AbstractTensor]] = None,
        loop=None,
        cert_path: str = None,
        key_path: str = None,
    ):

        # call WebsocketServerWorker constructor
        super().__init__(
            hook=hook,
            host=host,
            port=port,
            id=id,
            log_msgs=log_msgs,
            verbose=verbose,
            data=data,
            loop=loop,
            cert_path=cert_path,
            key_path=key_path
        )
        
        # self.object_store = CustomObjectStore(owner=self)
        # self.message_handlers.append(BaseMessageHandler(self.object_store, self))

        # Avoid Pytorch deadlock issues
        th.set_num_threads(1)

    # def fit(self, dataset_key: str, device: str = "cpu", **kwargs):
    #     """Fits a model on the local dataset as specified in the local TrainConfig object.
    #     Args:
    #         dataset_key: Identifier of the local dataset that shall be used for training.
    #         **kwargs: Unused.
    #     Returns:
    #         loss: Training loss on the last batch of training data.
    #     """
    #     self._check_train_config()

    #     if dataset_key not in self.datasets:
    #         raise ValueError("Dataset {} unknown.".format(dataset_key))

    #     model = self.get_obj(self.train_config._model_id).obj
    #     loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

    #     self._build_optimizer(
    #         self.train_config.optimizer, model, optimizer_args=self.train_config.optimizer_args
    #     )

    #     return self._fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn, device=device)

    # def _fit(self, model, dataset_key, loss_fn, device="cpu"):
    #     model.train()
    #     data_loader = self._create_data_loader(
    #         dataset_key=dataset_key, shuffle=self.train_config.shuffle
    #     )

    #     loss = None
    #     iteration_count = 0

    #     for _ in range(self.train_config.epochs):
    #         for (data, target) in data_loader:
    #             # Set gradients to zero
    #             self.optimizer.zero_grad()

    #             # Update model
    #             output = model(data.to(device))
    #             loss = loss_fn(target=target.to(device), pred=output)
    #             loss.backward()
    #             self.optimizer.step()

    #             # Update and check interation count
    #             iteration_count += 1
    #             if iteration_count >= self.train_config.max_nr_batches >= 0:
    #                 break

    #     return loss


    async def _producer_handler(self, websocket):
        """This handler listens to the queue and processes messages as they
        arrive.
        Args:
            websocket: the connection object we use to send responses
                back to the client.
        """
        while True:

            # logging.debug(f"Cummulated messages: {self.broadcast_queue}")

            # get a message from the queue
            message = await self.broadcast_queue.get()

            try:
                # convert that string message to the binary it represent
                message = binascii.unhexlify(message[2:-1])
            except binascii.Error as be:
                logging.debug(f"Erroneous message: {message}")
                logging.error(f"{be}")
                raise Exception

            # process the message
            response = self._recv_msg(message)

            # convert the binary to a string representation
            # (this is needed for the websocket library)
            response = str(binascii.hexlify(response))

            # send the response
            await websocket.send(response)


    def clear_residuals(self, exclusions: List=[]):
        """
        """
        logging.warning(f"Clear residuals triggered!")
        logging.warning(f"Before clearing - no. of objects: {len(self.object_store._objects)}")
        logging.warning(f"Exclusions recieved from TTP: {exclusions}")

        data_ids = [obj.id for obj in self.object_store.find_by_tag(tag="#X")]
        label_ids = [obj.id for obj in self.object_store.find_by_tag(tag="#y")]

        whitelisted_ids = data_ids + label_ids + exclusions
        logging.warning(f"Whitelisted IDs: {whitelisted_ids}")

        all_ids = list(self.object_store._objects.keys())
        for obj_id in all_ids:
            if obj_id not in whitelisted_ids:
                self.object_store.rm_obj(obj_id=obj_id)
            
        logging.warning(f"After clearing - no. of objects: {len(self.object_store._objects)}")

