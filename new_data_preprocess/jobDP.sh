#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --nodes=1

#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n16
source /opt/miniconda3.10/bin/activate

#SBATCH --partition=cuda
# set max wallclock time
#SBATCH --time=02:00:00
#SBATCH --output=dp_new_%j.log
#SBATCH --error=dp_new_%j.err

# set name of job
#SBATCH --job-name=dp_new


python new_data_preprocess/data_preprocess.py