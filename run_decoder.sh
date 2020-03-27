#!/bin/bash
#SBATCH --job-name=DGAT_decoder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:30:00
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=traian.vidrascu@student.uva.nl

conda activate dgat
python train_decoder.py --batch_size 128000  --dataset FB15k-237
python train_decoder.py --batch_size 128000  --dataset WN18RR
