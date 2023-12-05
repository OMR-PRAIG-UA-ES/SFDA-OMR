import random

import torch
from lightning.pytorch import LightningModule
from torch.nn import CTCLoss
from torchinfo import summary

from my_utils.augmentations import AugmentStage
from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import compute_all_metrics, ctc_greedy_decoder
from networks.modules import CRNN


class CTCTrainedCRNN(LightningModule):
    def __init__(
        self, w2i, i2w, use_augmentations=True, ytest_i2w=None, forbidden_chars=[]
    ):
        super(CTCTrainedCRNN, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.forbidden_chars = forbidden_chars
        # Model
        self.model = CRNN(output_size=len(self.w2i) + 1)
        self.summary()
        # Augmentations
        self.augment = AugmentStage() if use_augmentations else lambda x: x
        # Loss
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i), zero_infinity=True
        )  # The target index cannot be blank!
        # Predictions
        self.Y = []
        self.YHat = []

    def compute_forbidden_chars(self):
        # Characters that are in the test (target) vocabulary but not in the train (source) vocabulary
        # Train (source) vocabulary is the one known by the model
        self.forbidden_chars = [
            c for c in self.ytest_i2w.values() if c not in self.w2i.keys()
        ]

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, xl, y, yl = batch
        x = self.augment(x)
        yhat = self.model(x)
        # ------ CTC Requirements ------
        # yhat: [batch, frames, vocab_size]
        yhat = yhat.log_softmax(dim=2)
        yhat = yhat.permute(1, 0, 2).contiguous()
        # ------------------------------
        loss = self.compute_ctc_loss(yhat, y, xl, yl)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = ctc_greedy_decoder(yhat, self.i2w)
        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0]]
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        metrics = compute_all_metrics(self.Y, self.YHat, self.forbidden_chars)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
