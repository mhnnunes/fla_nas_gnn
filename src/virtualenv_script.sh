#!/bin/bash

# This script creates a virtual environment using Miniconda and 
# installs the dependencies necessary to run the code in this 
# repository. This script considers that the computer is using
# CUDA 10.1. If this is not your CUDA version, change
# the lines with comments highlighted below

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda install conda-build -y
conda create -y --name py38 python=3.8.3 && \
conda clean -ya
conda activate py38
conda install pip numpy mkl && \
conda clean -ya
conda install scipy pandas scikit-learn tqdm ipython -y && \
conda clean -ya
conda install -c conda-forge umap-learn -y && \
conda clean -ya
conda install -c pytorch magma-cuda101 && \
conda clean -ya
# IMPORTANT: if your computer is not using CUDA 10.1 the line below 
# has to contain a different version of magma-cuda.
# GO TO THIS WEBSITE AND PICK THE PYTORCH VERSION THAT COMPLIES WITH
# YOUR PYTHON, CUDA AND CUDNN VERSIONs:
# https://anaconda.org/pytorch/pytorch/files
# In this case it was cuda 10.1, python 3.8
conda install https://anaconda.org/pytorch/pytorch/1.6.0/download/linux-64/pytorch-1.6.0-py3.8_cuda10.1.243_cudnn7.6.3_0.tar.bz2 && \
conda clean -ya

TORCH=`python -c "import torch; print(torch.__version__)"`
CD=`python -c "import torch; print('cu' + str(torch.version.cuda.replace('.', ''))"`

pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CD}.html && \
pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CD}.html && \
pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CD}.html && \
pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CD}.html

pip install git+https://github.com/rusty1s/pytorch_geometric.git

pip install --no-cache-dir ogb==1.2.4