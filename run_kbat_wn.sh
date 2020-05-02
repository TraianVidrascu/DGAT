#!/bin/bash
#SBATCH --job-name=KBAT_wn18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python train_encoder.py --debug 0 --model KBAT --dataset WN18RR --margin 5 --epochs 3600 --step_size 500 --decay 5e-6