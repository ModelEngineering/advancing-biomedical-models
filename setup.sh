#!/bin/bash
# Assumes python3
# First install miniconda for python3.
# After running these installs do the following in the current
# directory:
#  jupyter notebook
#  open sample.ipynb
#  cell>run all
sudo $HOME/miniconda3/bin/pip install tellurium
sudo $HOME/miniconda3/bin/conda install jupyter notebook
sudo $HOME/miniconda3/bin/conda install scikit-learn
sudo $HOME/miniconda3/bin/conda install spyder
pip install lmfit
