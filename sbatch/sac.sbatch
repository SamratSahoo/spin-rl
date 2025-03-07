#!/bin/bash
#SBATCH -JSACTraining
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ssahoo61@gatech.edu

module load anaconda3/2023.03
source activate spin-rl
cd /home/hice1/ssahoo61/Documents/spin-rl
srun python main.py --model_config sac --env_config env