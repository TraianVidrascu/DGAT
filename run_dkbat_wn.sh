#!/bin/bash
#SBATCH --job-name=DKBAT_wn18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python train_encoder.py --debug 0 --model DKBAT --dataset WN18RR --margin 2 --epochs 3600 --step_size 500 --decay 5e-6