#!/bin/bash
# set name of job
#SBATCH --job-name=gin_hp
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n06
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=gin_hp_%j.log
#SBATCH --error=gin_%j.err




python new_training/hyperparam_gin.py