#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from datetime import datetime
from dateutil.tz import tzutc, tzlocal

# Libs


# Custom
from rest_rpc import app
from rest_rpc.core.datetime_serialization import DateTimeSerializer

##################
# Configurations #
##################

naive_date_string = '2020-09-14 03:30:10 N'
naive_date_obj = datetime(
    year=2020, month=9, day=14, hour=3, minute=30, second=10
)

tz_utc_date_string = '2020-09-14 03:30:10 UTC'
tz_utc_date_obj = datetime(
    year=2020, month=9, day=14, hour=3, minute=30, second=10, tzinfo=tzutc()
)

local_timezone = tzlocal().tzname(datetime.now())
tz_local_date_string = f'2020-09-14 03:30:10 {local_timezone}'
tz_local_date_obj = datetime(
    year=2020, month=9, day=14, hour=3, minute=30, second=10, tzinfo=tzlocal()
)

dt_serializer = DateTimeSerializer()

##################################
# DateTimeSerializer Class Tests #
##################################

def test_DateTimeSerializer_encode():
    naive_encoded_string = dt_serializer.encode(naive_date_obj)
    assert naive_encoded_string == naive_date_string
    tz_utc_encoded_string = dt_serializer.encode(tz_utc_date_obj)
    assert tz_utc_encoded_string == tz_utc_date_string
    tz_local_encoded_string = dt_serializer.encode(tz_local_date_obj)
    assert tz_local_encoded_string == tz_local_date_string


def test_DateTimeSerializer_decode():
    naive_decoded_obj = dt_serializer.decode(naive_date_string)
    assert naive_decoded_obj == naive_date_obj
    tz_utc_decoded_obj = dt_serializer.decode(tz_utc_date_string)
    assert tz_utc_decoded_obj == tz_utc_date_obj
    tz_local_decoded_obj = dt_serializer.decode(tz_local_date_string)
    assert tz_local_decoded_obj == tz_local_date_obj