#!/bin/bash
#SBATCH --job-name=DKBAT_fb15k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_encoder.py --debug 0 --model DKBAT --dataset FB15k-237 --margin 1 --epochs 2000 --step_size 250