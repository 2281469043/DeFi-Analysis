user_name=$(whoami)

pip install --upgrade pip --user

conda create --name $user_name-py36 python=3.6
conda activate $changk2-py36

pip install pyreadr
pip install ipykernel

python -m ipykernel install --changk2 --name=python36 --display=python36
