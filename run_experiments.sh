#!/bin/bash

# Baseline example (train and test):
for train_ds in "iam" "washington" "esposalles" "cvl"; do
    python -u train.py --ds_name "$train_ds"
    for test_ds in "iam" "washington" "esposalles" "cvl"; do
        python -u test.py --train_ds_name "$train_ds" --test_ds_name "$test_ds" --checkpoint_path "weights/Baseline-UpperBound/$train_ds.ckpt"
    done
done

# DA example:
python -u da_train.py --train_ds_name "iam" --test_ds_name "washington" --checkpoint_path "weights/Baseline-UpperBound/iam.ckpt" --bn_ids 1,5,9,13 --align_loss_weight 10 --minimize_loss_weight 10 --diversify_loss_weight 10 --lr 0.0003