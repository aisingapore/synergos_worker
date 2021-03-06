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
