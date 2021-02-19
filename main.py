#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import logging
import uuid

# Libs


# Custom
from rest_rpc import app

##################
# Configurations #
##################



###########
# Scripts #
###########

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="REST-RPC Receiver for a Synergos Network."
    )

    parser.add_argument(
        "--id",
        "-i",
        type=str,
        default=f"WRK_{uuid.uuid4}",
        help="ID of worker, e.g. --id Alice"
    )

    parser.add_argument(
        "--logging_variant",
        "-l",
        type=str,
        choices=["basic", "graylog"],
        default="basic",
        help="ID of worker, e.g. --id Alice"
    )

    parser.add_argument(
        "--id",
        "-i",
        type=str,
        required=True,
        help="ID of worker, e.g. --id Alice"
    )
    
    parser.add_argument(
        '--debug',
        "-d",
        action='store_true',
        default=False,
        help="Toggles debugging mode on this worker node"
    )

    kwargs = vars(parser.parse_args())
    logging.info(f"Worker Parameters: {kwargs}")

    app.run(host="0.0.0.0", port=5000)



##############
# Deprecated #
##############
"""
"""