#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --nodes=1

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=02:00:00
#SBATCH --output=gcn_hp_%j.log
#SBATCH --error=gcn_%j.err

# set name of job
#SBATCH --job-name=gcn_hp_training


python new_training/hyperparam_gcn.py