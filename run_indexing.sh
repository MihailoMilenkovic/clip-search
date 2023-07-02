#!/bin/bash
#SBATCH --job-name=clip-indexing-test
#SBATCH --ntasks=1
#SBATCH --nodelist=n16
#SBATCH --partition=cuda
#SBATCH --time=24:00:00

python indexing_example.py