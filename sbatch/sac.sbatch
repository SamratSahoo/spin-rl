#!/bin/bash
#SBATCH -JSACTrainingGPU
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out

module load anaconda3/2023.03
source activate spin-rl
cd /home/hice1/ssahoo61/Documents/spin-rl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
srun python main.py --algorithm sac