#!/bin/bash
#SBATCH --job-name=clip-training-test
#SBATCH --ntasks=1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
#SBATCH --time=24:00:00

python clip_model.py