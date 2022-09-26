#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=32gb
#PBS -lwalltime=08:00:00

export PATH=$HOME/miniconda3/bin/:$PATH
export HOME=$HOME
export PATH_INTP_FOLDER=$HOME/ICLR_Interp

source activate
conda activate interp
jupyter notebook --no-browser --port 1234