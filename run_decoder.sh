#!/bin/bash
#SBATCH --job-name=DGAT_decoder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:30:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_decoder.py --batch_size 64000 --eval 25 --epochs 200 --dataset FB15k-237
python train_decoder.py --batch_size 64000 --eval 25 --epochs 200 --dataset WN18RR
