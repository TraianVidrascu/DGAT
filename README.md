# Bidirectional Neighborhood Model (Encoder Only)

The Bidirectional Neighborhood Model is an adaptation of the KBAT model from the paper: [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://arxiv.org/abs/1906.01195)

This implementation is standalone, but parts of the code are taken from the [KBAT pipeline](https://github.com/deepakn97/relationPrediction).

# Installation
Use the [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) package manager to install the environment of this project.

```bash
conda env create -f envname.yml
source activate base
``` 

# Modifications
- usage of the inbound and outbound node neighborhoods 
- Topological Relation Layer

# Datasets
- FB15k-237
- WN18RR
- Kinship
- Ivy v1.4.1 software graph (Private company dataset)

# Reproduce Results (Encoder Only)
- FB15k-237 
```bash
python train_encoder.py --backprop_entity 0 --backprop_relation 1 --eval 1000  --use_paths 0 --use_partial 0 --debug 0 --model DKBAT --dataset FB15K-237 --margin 1 --output_encoder 200 --batch -1 --negative_ratio 2 --epochs 3000 --step_size 250
``` 

- WN18RR
```bash
python train_encoder.py  --backprop_entity 0 --backprop_relation 1 --eval 1000  --use_paths 0 --use_partial 0 --debug 0 --model DKBAT --dataset WN18RR --margin 5 --output_encoder 200 --batch -1 --negative_ratio 2 --epochs 3000 --step_size 250 --decay 5e-6
``` 

- Kinship
```bash
pyton train_decoder.py --output_encoder 400 --model DKBAT --eval 100 --debug 0 --margin 1 --batch -1 --dataset KINSHIP
``` 

# License
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/) 
