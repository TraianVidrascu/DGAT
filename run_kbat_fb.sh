#!/bin/bash
#SBATCH --job-name=KBAT_fb15k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_encoder.py --debug 0 --use_paths 1 --use_partial 1 --epochs 3000 --step_size 500 True --model KBAT --dataset FB15K-237 --margin 1 --batch -1