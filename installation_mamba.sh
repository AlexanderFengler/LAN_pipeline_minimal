#!/bin/bash

# Note: this assumes you have mamba installed
source ~/.bashrc

conda deactivate 
conda deactivate
conda remove -n lan-pipeline-test --all
mamba create -n lan-pipeline-test python=3.11
conda activate lan-pipeline-test

pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lanfactory
pip install ssm-simulators
pip install jupyter
pip install frozendict
pip install pyyaml
pip install wandb

python -m ipykernel install --user --name=lan-pipeline-test