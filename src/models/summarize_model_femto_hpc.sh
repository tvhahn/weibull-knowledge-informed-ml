#!/bin/bash
#SBATCH --account=rrg-mechefsk
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=12G      # memory per node
#SBATCH --time=0-00:20      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

module load python/3.8
source ~/weibull/bin/activate

python $PROJECT_DIR/src/models/summarize_model_results.py --data_set femto
