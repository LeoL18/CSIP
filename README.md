# CSIP

This repository contains the implementation of CSIP and can be used to reproduce the results in (link). The implementation is based on an open source implementation of OpenAI's CLIP (OpenAI link). 

## Abstract


## Data
The dataset we used can be found at (link), and the scripts for downloading the dataset can be found at (link). Preprocessed image profiles and metadata can be found at (link).

## Setup


## Usage
python train.py \ 
--load_dir "<path to dataset>" \
--load_dir_label “D:\Cell painting” \
--save_dir "output" \
--device "cpu" \
--use_wandb 0 \
--debug 0 \
--learnable_tau 1 \
--lr 
--loss "standard" 

## Retrieval task
We provide a Jupyter notebook (link) for a demonstration of molecule retrieval with CSIP. 
