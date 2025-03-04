#!/bin/bash
#SBATCH -JSACTraining
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ssahoo61@gatech.edu

cd /home/hice1/ssahoo61/Documents/spin-rl
source ../venv/bin/activate
srun python main.py --model_config sac --env_config env