#!/bin/bash
#SBATCH --job-name=KBAT_ConvKB_wn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=27:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl


conda activate dgat
python train_decoder.py --debug 0 --model KBAT --batch_size 128 --channels 500 --dropout 0 --negative-ratio 40 --eval 100 --epochs 200 --dataset WN18RR

