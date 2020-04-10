#!/bin/bash
#SBATCH --job-name=DKBAT_ConvKB_fb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=5:00:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_decoder.py --debug 0 model DKBAT --batch_size 128 --negative-ratio 40 --eval 200 --epochs 200 --dataset FB15k-237