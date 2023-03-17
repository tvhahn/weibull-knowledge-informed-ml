#!/bin/bash
conda install -n base -c conda-forge mamba
mamba env create -f envweibull.yml # install the environment with conda (use mamba)
eval "$(conda shell.bash hook)"
conda activate weibull # activate the environment
pip install -e .