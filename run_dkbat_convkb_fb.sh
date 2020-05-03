#!/bin/bash
#SBATCH --job-name=DKBAT_ConvKB_fb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=27:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_decoder.py --debug 0 --model DKBAT --channels 50  --dropout 0.3 --batch_size 128 --negative-ratio 40 --eval 75 --epochs 150 --dataset FB15K-237