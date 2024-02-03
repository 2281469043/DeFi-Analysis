#!/bin/bash

# Get user name
user_name=$(whoami)

pip install --upgrade pip --user

conda create --name $user_name-py36 python=3.6
conda activate $user_name-py36

pip install pyreadr
pip install ipykernel

python -m ipykernel install --user --name=python36 --display=python36
