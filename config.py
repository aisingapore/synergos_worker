#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import os
from collections import defaultdict
from pathlib import Path

# Libs
from tinydb import TinyDB, Query
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

##################
# Configurations #
##################

SRC_DIR = Path(__file__).parent.absolute()

db_path = os.path.join(SRC_DIR, "database.json")

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

###############################
# PySyft Local Configurations #
###############################

server_params = {
    
    # Define server's role: Master or slave
    'is_master': False,
    
    # State location of data
    'data_dir': os.path.join(SRC_DIR, "data"),

    # State output directory
    'out_dir': os.path.join(SRC_DIR, "outputs"),

    # Initialise database
    'database': TinyDB(
        path=db_path, 
        sort_keys=True,
        indent=4,
        separators=(',', ': '),
        default_table="Cache",
        storage=CachingMiddleware(JSONStorage)
    ),

    # Initialise Cache
    'cache': infinite_nested_dict()

}