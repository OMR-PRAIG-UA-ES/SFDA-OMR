import gc
import random

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from data.synthetic_words_random.dataset import CTCSyntheticRandomDataset
from data.synthetic_words_real.dataset import CTCSyntheticRealDataset
from my_utils.data_preprocessing import (
    ctc_batch_preparation,
    multisource_ctc_batch_preparation,
)
from my_utils.dataset import (
    DS_CONFIG,
    CTCDataset,
    TestMultiSourceCTCDataset,
    TrainMultiSourceCTCDataset,
)
from networks.model import CTCTrainedCRNN


def train(
    ds_name,
    epochs,
    patience,
    batch_size,
    use_augmentations=True,
    lowercase=False,
    metric_to_monitor="val_cer",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check dataset(s)
    ds_name = ds_name.split(" ")
    if not all([ds in DS_CONFIG.keys() for ds in ds_name]):
        raise NotImplementedError("Dataset not implemented")

    # Get dataset(s)
    if len(ds_name) > 1:
        if lowercase:
            # Lowercase should only be used in single-source setting
            # because the synthetic dataset IIIIT-HWS is lowercase
            # We want to compare the performance of a synthetic-trained (IIIT-HWS) model
            # on real data (IAM, GW, Espossales, CVL)
            # Not an implementation error per se (more like a warning)
            raise NotImplementedError(
                "Lowercase should only be used in single-source setting"
            )

        # MULTI-SOURCE SETTING
        train_ds = TrainMultiSourceCTCDataset(
            name="_".join(ds_name),
            img_folder_paths=[DS_CONFIG[ds]["train"] for ds in ds_name],
            transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_name],
        )
        val_ds = TestMultiSourceCTCDataset(
            name="_".join(ds_name),
            img_folder_paths=[DS_CONFIG[ds]["val"] for ds in ds_name],
            transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_name],
            w2i=train_ds.w2i,
            i2w=train_ds.i2w,
        )
        test_ds = TestMultiSourceCTCDataset(
            name="_".join(ds_name),
            img_folder_paths=[DS_CONFIG[ds]["test"] for ds in ds_name],
            transcripts_files=[DS_CONFIG[ds]["transcripts"] for ds in ds_name],
            w2i=train_ds.w2i,
            i2w=train_ds.i2w,
        )
        batch_size = batch_size // len(ds_name)
        ds_name = "_".join(ds_name)
        collate_fn = multisource_ctc_batch_preparation

    else:
        # SINGLE-SOURCE SETTING
        ds_name = "_".join(ds_name)

        if ds_name == "synthetic_words_random":
            train_ds = CTCSyntheticRandomDataset(
                name=ds_name,
                transcripts_file=DS_CONFIG[ds_name]["train_transcripts"],
                samples_per_epoch=DS_CONFIG[ds_name]["train"],
                lowercase=lowercase,
            )
        elif ds_name == "synthetic_words_real" or ds_name == "synthetic_words_wiki":
            train_ds = CTCSyntheticRealDataset(
                name=ds_name,
                transcripts_file=DS_CONFIG[ds_name]["train_transcripts"],
                lowercase=lowercase,
            )
        else:
            train_ds = CTCDataset(
                name=ds_name,
                img_folder_path=DS_CONFIG[ds_name]["train"],
                transcripts_file=DS_CONFIG[ds_name]["transcripts"],
                lowercase=lowercase,
            )
        val_ds = CTCDataset(
            name=ds_name,
            img_folder_path=DS_CONFIG[ds_name]["val"],
            transcripts_file=DS_CONFIG[ds_name]["transcripts"],
            lowercase=lowercase,
            train=False,
        )
        if "synthetic_words" not in ds_name:
            test_ds = CTCDataset(
                name=ds_name,
                img_folder_path=DS_CONFIG[ds_name]["test"],
                transcripts_file=DS_CONFIG[ds_name]["transcripts"],
                lowercase=lowercase,
                train=False,
            )
        collate_fn = ctc_batch_preparation

    # Experiment info
    print("Running experiment: Baseline-UpperBound")
    print(f"\tDataset(s): {ds_name}")
    print(f"\tLowercase: {lowercase}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        collate_fn=collate_fn,
    )  # prefetch_factor=2
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2
    if "synthetic_words" not in ds_name:
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=20
        )  # prefetch_factor=2

    # Model
    model = CTCTrainedCRNN(
        w2i=train_ds.w2i, i2w=train_ds.i2w, use_augmentations=use_augmentations
    )
    train_ds.width_reduction = model.model.cnn.width_reduction
    model_name = ds_name
    model_name += "_lowercase" if lowercase else ""

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath="weights/Baseline-UpperBound",
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
            project="Unsupervised-Adaptation-Word-HTR",
            group="Baseline-UpperBound",
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
    if "synthetic_words" not in ds_name:
        model = CTCTrainedCRNN.load_from_checkpoint(
            f"weights/Baseline-UpperBound/{model_name}.ckpt", ytest_i2w=test_ds.i2w
        )
        model.freeze()
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
