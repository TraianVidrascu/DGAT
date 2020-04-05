#!/bin/bash
#SBATCH --job-name=DKBAT_ConvKB_wn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:30:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_decoder.py --debug 0 --model DKBAT --channels 500 --dropout 0 --batch_size 128 --negative-ratio 40 --eval 200 --epochs 200 --dataset WN18RR