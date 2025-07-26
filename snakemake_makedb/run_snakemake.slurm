#!/bin/bash
#SBATCH --job-name snakemake_makedb
#SBATCH -p sched_mit_tami
#SBATCH -n 1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH -o mainout.txt
#SBATCH -e mainerr.txt
#SBATCH --mail-user=YOUR_EMAIL_HERE
#SBATCH --mail-type=ALL

# Activate conda environment (may need to change name of env)
source activate snakemake

snakemake --profile .

# Print "Done!!!" at end of main log file
echo Done!!!
