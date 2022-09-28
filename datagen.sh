#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=32gb
#PBS -lwalltime=06:00:00
#PBS -J 1-2

export PATH=$HOME/miniconda3/bin/:$PATH
export HOME=$HOME
export PATH_INTP_FOLDER=$HOME/ICLR_Interp

source activate
conda activate interp
python $HOME/ICLR_Interp/DataGeneration/DataGen.py
