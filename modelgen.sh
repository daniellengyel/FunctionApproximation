#!/bin/bash
#PBS -lselect=1:ncpus=2:mem=64gb
#PBS -lwalltime=08:00:00
##PBS -J 1-594

export PATH=$HOME/miniconda3/bin/:$PATH
export HOME=$HOME
export PATH_INTP_FOLDER=$HOME/ICLR_Interp

source activate
conda activate interp
python $HOME/ICLR_Interp/ModelGeneration/nn_generation.py

