#!/bin/bash
#SBATCH --job-name=CMOR-421-521
#SBATCH --partition=scavenge
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:30:00

echo "My job ran on:"
ech $SLURM_NODELIST

srun -n 2 ./hello