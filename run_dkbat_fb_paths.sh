#!/bin/bash
#SBATCH --job-name=DKBAT_fb15k_paths
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_encoder.py --eval 1000  --use_paths 1 --use_partial 1 --debug 0 --model DKBAT --dataset FB15K-237 --margin 1 --output_encoder 200 --batch 2000 --negative_ratio 4 --epochs 1000 --step_size 250