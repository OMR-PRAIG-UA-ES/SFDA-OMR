import gc
import os
import random

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from my_utils.dataset import DS_CONFIG, CTCDataset
from networks.model import CTCTrainedCRNN


def test(
    train_ds_name,
    test_ds_name,
    checkpoint_path,
    lowercase=False,
    project="Unsupervised-Adaptation-Word-HTR",
    group="Baseline-LowerBound",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if multi-source setting
    train_ds_name = "_".join(train_ds_name.split(" "))

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Experiment info
    print("Running experiment: Baseline-LowerBound")
    print(f"\tTest dataset: {test_ds_name}")
    print(f"\tCheckpoint path ({train_ds_name}): {checkpoint_path}")
    print(f"\tLowercase: {lowercase}")

    # Dataset
    if test_ds_name in DS_CONFIG.keys() and "synthetic_words" not in test_ds_name:
        test_ds = CTCDataset(
            name=test_ds_name,
            img_folder_path=DS_CONFIG[test_ds_name]["test"],
            transcripts_file=DS_CONFIG[test_ds_name]["transcripts"],
            lowercase=lowercase,
            train=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=20
        )  # prefetch_factor=2
    else:
        raise NotImplementedError("Dataset not implemented")

    # Model
    model = CTCTrainedCRNN.load_from_checkpoint(checkpoint_path, ytest_i2w=test_ds.i2w)
    model.freeze()

    # Compute forbidden characters
    model.compute_forbidden_chars()
    print(f"\tForbidden characters: {model.forbidden_chars}")

    # Test: automatically auto-loads the best weights from the previous run
    run_name = f"Train-{train_ds_name}_Test-{test_ds_name}"
    run_name += "_lowercase" if lowercase else ""
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=run_name,
            log_model=False,
        ),
        precision="16-mixed",
    )
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(test)
