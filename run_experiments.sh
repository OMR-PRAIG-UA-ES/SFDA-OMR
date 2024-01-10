#!/bin/bash

# BASELINE
# - MENSURAL
for encoding in standard split; do
    for train_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
        for test_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
            if [ $train_ds != $test_ds ]; then
                python -u test.py --train_ds_name $train_ds --test_ds_name $test_ds --checkpoint_path weights/Baseline-UpperBound/$train_ds\_$encoding.ckpt --encoding_type $encoding
            fi
        done
    done
done
# - MODERN
for encoding in standard split; do
    for train_ds in Primus CameraPrimus FMT Malaga; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
        for test_ds in Primus CameraPrimus FMT Malaga; do
            if [ $train_ds != $test_ds ]; then
                python -u test.py --train_ds_name $train_ds --test_ds_name $test_ds --checkpoint_path weights/Baseline-UpperBound/$train_ds\_$encoding.ckpt --encoding_type $encoding
            fi
        done
    done
done

# AMD SOURCE-FREE DOMAIN ADAPTATION
# DA example:
python -u da_train.py --train_ds_name b-59-850 --test_ds_name ILS --checkpoint_path weights/Baseline-UpperBound/b-59-850_standard.ckpt --bn_ids 1,5,9,13 --align_loss_weight 10 --minimize_loss_weight 10 --diversify_loss_weight 10 --lr 0.0003 --encoding_type standard
