#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=gcn_train_new
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=02:00:00
#SBATCH --output=gcn_%j.log
#SBATCH --error=gcn_%j.err



python new_training/gcn_train.py