import gc
import random

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from my_utils.data_preprocessing import ctc_batch_preparation
from my_utils.dataset import DS_CONFIG, CTCDataset
from networks.model import CTCTrainedCRNN

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(
    ds_name,
    epochs,
    patience,
    batch_size,
    use_augmentations=True,
    metric_to_monitor="val_ser",
    project="AMD-OMR",
    group="Baseline-UpperBound",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if dataset exists
    if not ds_name in DS_CONFIG.keys():
        raise NotImplementedError(f"Dataset {ds_name} not implemented")

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tDataset(s): {ds_name}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    # Get datasets
    train_ds = CTCDataset(
        name=ds_name,
        img_folder_path=DS_CONFIG[ds_name]["train"],
        transcripts_file=DS_CONFIG[ds_name]["transcripts"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        collate_fn=ctc_batch_preparation,
    )  # prefetch_factor=2
    val_ds = CTCDataset(
        name=ds_name,
        img_folder_path=DS_CONFIG[ds_name]["val"],
        transcripts_file=DS_CONFIG[ds_name]["transcripts"],
        train=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2
    test_ds = CTCDataset(
        name=ds_name,
        img_folder_path=DS_CONFIG[ds_name]["test"],
        transcripts_file=DS_CONFIG[ds_name]["transcripts"],
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2

    # Model
    model = CTCTrainedCRNN(
        w2i=train_ds.w2i, i2w=train_ds.i2w, use_augmentations=use_augmentations
    )
    train_ds.width_reduction = model.model.cnn.width_reduction
    model_name = ds_name

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=model_name,
            monitor=metric_to_monitor,
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=metric_to_monitor,
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=f"Train-{ds_name}_Test-{model_name}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    model = CTCTrainedCRNN.load_from_checkpoint(
        f"weights/{group}/{model_name}.ckpt", ytest_i2w=test_ds.i2w
    )
    model.freeze()
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
