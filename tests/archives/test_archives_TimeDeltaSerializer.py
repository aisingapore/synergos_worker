#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from datetime import timedelta


# Libs


# Custom
from rest_rpc import app
from rest_rpc.core.datetime_serialization import TimeDeltaSerializer

##################
# Configurations #
##################

timedelta_string = "20.20" # 20 days + 20 seconds
timedelta_obj = timedelta(days=20, seconds=20)

td_serializer = TimeDeltaSerializer()

###################################
# TimeDeltaSerializer Class Tests #
###################################

def test_TimeDeltaSerializer_encode():
    timedelta_encoded_string = td_serializer.encode(timedelta_obj)
    assert timedelta_encoded_string == timedelta_string


def test_TimeDeltaSerializer_decode():
    timedelta_decoded_obj = td_serializer.decode(timedelta_string)
    assert timedelta_decoded_obj == timedelta_obj