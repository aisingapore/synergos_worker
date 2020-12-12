#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import logging

# Libs


# Custom
from rest_rpc import app

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

###########
# Scripts #
###########

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


##############
# Deprecated #
##############
"""
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
