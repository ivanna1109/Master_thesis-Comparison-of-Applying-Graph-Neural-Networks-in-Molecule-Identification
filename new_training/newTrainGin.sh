#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=gin_tr
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n16
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=02:00:00
#SBATCH --output=gin_%j.log
#SBATCH --error=gin_%j.err



python new_training/gin_train.py