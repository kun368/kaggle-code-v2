#!/usr/bin/env bash

cd /mnt/d/Code/Pycharm/kaggle-code-v2 || exit
source activate
conda activate py37
jupyter notebook
