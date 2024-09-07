#!/bin/bash
# set name of job
#SBATCH --job-name=gsage_hp
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n16
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=gsage_hp_%j.log
#SBATCH --error=gsage_%j.err


python new_training/hyperparam_gSage.py