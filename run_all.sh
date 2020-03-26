#!/bin/bash
#SBATCH --job-name=DGAT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:30:00
#SBATCH --mem=90000M
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_encoder.py
python train_decoder.py  --batch_size 64000
