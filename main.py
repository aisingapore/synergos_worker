#!/usr/bin/env python

####################
# Required Modules #
####################

# from gevent import monkey
# monkey.patch_all()

# Generic/Built-in
import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path

# Libs
# from gevent.pywsgi import WSGIServer
# from gunicorn.app.base import BaseApplication

# Custom
from rest_rpc import app

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

app.secret_key = "synergos_worker" #os.urandom(24) # secret key

cores_used = app.config['CORES_USED']

cache = app.config['CACHE']

##################
# Customisations #
##################

def number_of_workers():
    return (cores_used * 2) + 1


# class SynergosWorkerNode(BaseApplication):

#     def __init__(self, app, **options):
#         self.options = options
#         self.application = app
#         super().__init__()

#     def load_config(self):
#         config = {
#             key: value
#             for key, value in self.options.items()
#             if key in self.cfg.settings and value is not None
#         }
#         for key, value in config.items():
#             self.cfg.set(key.lower(), value)

#     def load(self):
#         return self.application

###########
# Scripts #
###########

if __name__ == "__main__":
    
    cache.init_app(
        app=app, 
        config={
            "CACHE_TYPE": "filesystem",
            'CACHE_DIR': Path('/worker/tmp')
        }
    )

    app.run(host="0.0.0.0", port=5000)


    # # worker_server = WSGIServer(('0.0.0.0', 5000), app)
    # # worker_server.serve_forever()

    # DEFAULTS = {
    #     'bind': "0.0.0.0:5000",
    #     'name': "synergos_worker"
    # }

    # parser = argparse.ArgumentParser(
    #     description="Synergos Worker Node for grid orchestration."
    # )

    # parser.add_argument(
    #     "--workers",
    #     "-w",
    #     type=int,
    #     help="No. of workers to spawn for handling requests e.g. --workers 2",
    #     default=1#number_of_workers()
    # )

    # parser.add_argument(
    #     "--worker-class",
    #     "-k",
    #     type=str,
    #     help="Type of Gunicorn worker to be spawned i.e. sync, eventlet, gevent, tornado, gthread",
    #     default="sync"
    # )

    # parser.add_argument(
    #     "--timeout",
    #     "-t",
    #     type=int,
    #     help="Duration of silence before a worker is killed and restarted",
    #     default=30
    # )

    # parser.add_argument(
    #     "--threads",
    #     "-tds",
    #     type=int,
    #     help="No. of threads to use per worker",
    #     default=1
    # )

    # parser.add_argument(
    #     "--preload",
    #     "-p",
    #     help="Toggles loading of application code before worker processes are forked",
    #     action='store_true'
    # )

    # parser.add_argument(
    #     "--debug",
    #     "-d",
    #     help="Toggles loading of application code before worker processes are forked",
    #     action='store_true'
    # )

    # kwargs = vars(parser.parse_args())

    # gunicorn_parameters = {**DEFAULTS, **kwargs}
    # logging.info(f"Gunicorn settings: {gunicorn_parameters}")

    # server = SynergosWorkerNode(app=app, **gunicorn_parameters)
    # server.run()
