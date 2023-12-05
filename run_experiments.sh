#!/bin/bash

######################### SINGLE-SOURCE EXPERIMENTS:

# 1) Get upper bound (train and test on the same dataset) and lower bound (train on one dataset and test on another):

for train_ds in "iam" "washington" "esposalles" "cvl"; do
    python -u train.py --ds_name "$train_ds" --epochs 1000 --patience 20 --batch_size 16
    for test_ds in "iam" "washington" "esposalles" "cvl"; do
        python -u test.py --train_ds_name "$train_ds" --test_ds_name "$test_ds" --checkpoint_path "weights/Baseline-UpperBound/$train_ds.ckpt"
    done
done

######################### MULTI-SOURCE EXPERIMENTS:

# 1) Get upper bound (train and test on the same dataset) and lower bound (train on one dataset and test on another):
comb_train_ds=("iam washington esposalles" "iam washington cvl" "iam esposalles cvl" "washington esposalles cvl")
for train_ds in "${comb_train_ds[@]}"; do
    python -u train.py --ds_name "$train_ds" --epochs 1000 --patience 20 --batch_size 16
    for test_ds in "iam" "washington" "esposalles" "cvl"; do
        python -u test.py --train_ds_name "$train_ds" --test_ds_name "$test_ds" --checkpoint_path "weights/Baseline-UpperBound/${train_ds// /_}.ckpt"
    done
done