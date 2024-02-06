#!/bin/bash

# This script is used to create a conda environment for our project
# let us successfully run the codes in the project server.

# Get user name here
user_name=$(whoami)

conda init

source .bashrc

pip install --upgrade pip --user

conda create --name $user_name-py36 python=3.6
conda activate $user_name-py36

pip install pyreadr
pip install ipykernel

python -m ipykernel install --user --name=python36 --display=python36