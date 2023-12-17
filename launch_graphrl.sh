#!/bin/bash

#SBATCH -A vascnetgen # Project name
#SBATCH -n 50 # Number of cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --mem=7000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o log.out # File to which STDOUT will be written
#SBATCH -e log.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jbsimoes@dei.uc.pt # Email to which notifications will be sent
#SBATCH --gres=gpu:1 # Number of GPUs per node

# Run the program and pass the arguments from the command line
poetry run python /veracruz/projects/v/vascnetgen/graph-rl/main.py
