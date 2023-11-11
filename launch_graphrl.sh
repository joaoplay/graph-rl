#!/bin/bash

#SBATCH -A vascnetgen # Project name
#SBATCH -n 30 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p cpu2 # Partition to submit to
#SBATCH --mem=70000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o log.out # File to which STDOUT will be written
#SBATCH -e log.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jbsimoes@dei.uc.pt # Email to which notifications will be sent

poetry run /veracruz/projects/v/vascnetgen/graph-rl/main.py