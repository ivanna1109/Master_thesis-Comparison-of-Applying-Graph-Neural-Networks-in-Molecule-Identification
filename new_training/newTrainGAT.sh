#!/bin/bash
# set name of job
#SBATCH --job-name=gat_train
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n06
source /opt/miniconda3.10/bin/activate

#SBATCH --time=02:00:00
#SBATCH --output=gat_hp_%j.log
#SBATCH --error=gat_%j.err


python new_training/gat_train.py