#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from datetime import datetime, timedelta
from dateutil.tz import tzutc, tzlocal

# Libs
from tinydb_serialization import Serializer

############################################
# Serialization Class - DateTimeSerializer #
############################################

class DateTimeSerializer(Serializer):
    """
    Serialises both aware and naive datetime objects for storage into TinyDB.

    Encoding - If the datetime object is aware, it is first converted to UTC and
               then encoded with an 'A' appended to the serialization. Otherwise
               it is serialized without conversion and an 'N' is appended.
    Decoding - If the serialization ends with 'A', the datetime object is 
               treated as UTC and then converted to localtime. Otherwise, the 
               datetime object is treated as localtime and no conversion is 
               necessary.

    Note: Microseconds are discarded but hours, minutes & seconds are preserved.
    """
    OBJ_CLASS = datetime

    def encode(self, obj: datetime) -> str:
        """ Serialize naive datetimes objects without conversion but with 'N' 
            for 'Naive' appended. Convert aware datetime objects to UTC if their
            timezone specified was UTC, otherwise convert them to the detected
            local timezone.

        Args:
            obj (datetime.datetime): Datetime object to be encoded
        Returns:
            Serialized datetime string (str)
        """
        if obj.tzinfo is None:
            return obj.strftime('%Y-%m-%d %H:%M:%S N')
        else:
            return obj.strftime('%Y-%m-%d %H:%M:%S %Z')

    def decode(self, s: str) -> str:
        """ Return the serialization as a datetime object. If the serialization 
            ends with 'N', return without conversion as a naive datetime object.
            However, if a timezone is detected, serialisation is returned as
            UTC casted if the UTC timezone detected, otherwise serialisation is
            returned with local timezone casted

        Args:
            s (str): Serialized datetime string
        Returns:
            Decoded datetime object
        """
        if s[-1] == 'N': 
            # Handle naive cases
            return datetime.strptime(s[:-2], '%Y-%m-%d %H:%M:%S')
        
        elif s[-3:] == 'UTC':
            # Handle UTC timezone
            return datetime.strptime(
                s, '%Y-%m-%d %H:%M:%S %Z'
            ).replace(tzinfo=tzutc())
            
        else: 
            # Handle local timezones
            return datetime.strptime(
                s, '%Y-%m-%d %H:%M:%S %Z'
            ).replace(tzinfo=tzlocal())


class TimeDeltaSerializer(Serializer):
    """
    Handles serialisations of timedeltas (for checking datelines)
    """
    OBJ_CLASS = timedelta  # The class handles timedelta objects

    def encode(self, obj: timedelta) -> str:
        """ Serialize the timedelta object as '{days}.{seconds}'. 
        
        Args:
            obj (datetime.timedelta): Timedelta object to be serialized
        Returns:
            Serialized timedelta string (str)
        """
        return "{0}.{1}".format(obj.days, obj.seconds)

    def decode(self, s: str) -> timedelta:
        """ Return the serialization as a timedelta object 
        
        Args:
            s (str): Serialized timedelta string
        Returns:
            Decoded timedelta object
        """
        days_seconds = (int(x) for x in s.split('.')) 
        return timedelta(*days_seconds)