#!/bin/bash
#SBATCH --job-name=DKBAT_wn18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python train_encoder.py --eval 1000  --use_paths 0 --use_partial 0 --debug 0 --model DKBAT --dataset WN18RR --margin 5 --batch 2000 --negative_ratio 4 --epochs 1000 --step_size 250 --decay 5e-6