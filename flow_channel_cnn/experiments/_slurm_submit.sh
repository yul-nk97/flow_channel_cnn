#!/bin/bash
#SBATCH --job-name=flow_cnn
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=90G

# Activate virtual environment if needed
source /media/ssd/Programming/flow_channel_cnn/venv/bin/activate

# Run your script
python "/media/ssd/Programming/flow_channel_cnn/flow_channel_cnn/experiments/train_invariant_cnn.py" \
    --__DEBUG__="False" 
