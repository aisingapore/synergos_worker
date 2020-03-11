#!/usr/bin/env python

####################
# Required Modules #
####################

# Generics
from subprocess import Popen

##########
# Script #
##########

if __name__ == "__main__":


    # Pyenv Prerequisites
    "sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git"

    # Installing Pyenv + Virtualenv plugin
    curl https://pyenv.run | bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    # Refresh the shell
    exec $SHELL

    # Install Supported Python version
    pyenv install 


    protocol.tag('horizontal')