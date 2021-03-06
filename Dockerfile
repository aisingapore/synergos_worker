# FROM python:3.7.4-slim-buster

# ##################
# # Configurations #
# ##################

# ARG PYSYFT_CHECKPOINT=PySyft-v0.2.4_aisg
# ARG SYFT_PROTO_CHECKPOINT=syft-proto-v0.2.5.a1_aisg

# ##########################################################
# # Step 1: Install Linux dependencies for source-building #
# ##########################################################

# RUN apt-get update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         build-essential \
#         libz-dev \
#         libbz2-dev \
#         libc6-dev \
#         libdb-dev \
#         libgdbm-dev \
#         libffi-dev \
#         liblzma-dev \
#         libncursesw5-dev \
#         libreadline-gplv2-dev \
#         libsqlite3-dev \
#         libssl-dev \
#         tk-dev \
#         git \
#         unzip

# #########################################
# # Step 2: Clone, build & setup REST-RPC #
# #########################################

# ADD . /worker
# WORKDIR /worker

# RUN pip install --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir -r requirements.txt

# RUN unzip -q '/worker/etc/*.zip' -d /worker/etc/tmp \
#  && pip install /worker/etc/tmp/${PYSYFT_CHECKPOINT} \
#  && pip install /worker/etc/tmp/${SYFT_PROTO_CHECKPOINT}

# #######################################################
# # Step 3: Expose relevant connection points & run app #
# #######################################################

# EXPOSE 5000
# EXPOSE 8020

# ENTRYPOINT ["python", "./main.py"]

# CMD ["--help"]

##############
# Base Image #
##############

FROM python:3.7.4-slim-buster as base

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git\
    pciutils

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ADD . /worker
WORKDIR /worker

RUN pip install ./synergos_logger

EXPOSE 5000
EXPOSE 8020

################################################
# Implementation Footnote - Poetry Integration #
################################################

# Problem:
# Poetry is a deterministic way of managing dependency installations. However,
# there seems to be trouble in installing Tensorflow, as documented under issue
# #1331 (https://github.com/python-poetry/poetry/issues/1330). 
# Solution: K.I.V until issue is resolved

# FROM python:3.7-slim as base

# RUN apt-get update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git python3-pip

# RUN pip install --upgrade setuptools wheel poetry

# ADD . /worker
# WORKDIR /worker

# RUN poetry install

# # Quick fix for tensorflow installation with Poetry
# RUN poetry add tensorflow@1.15.0 tf-encrypted@0.5.9

########################
# New Image - Debugger #
########################

FROM base as debug
RUN pip install ptvsd

WORKDIR /worker
EXPOSE 5678
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait main.py

##########################
# New Image - Production #
##########################

FROM base as prod

WORKDIR /worker
ENTRYPOINT ["python", "./main.py"]
CMD ["--help"]
