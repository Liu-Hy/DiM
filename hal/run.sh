#!/bin/bash
#SBATCH --job-name="single_node"
#SBATCH --output="demo.%j.%N.out"
#SBATCH --error="demo.%j.%N.err"
#SBATCH --partition=gpux2

module load conda_base
conda activate new_env
python python dg.py --data PACS  --match-aug