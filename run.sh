#!/bin/bash
#SBATCH --job-name=DGAT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train.py
