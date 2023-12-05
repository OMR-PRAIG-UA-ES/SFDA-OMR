import torch
import torch.nn as nn


class DomainAdaptationLoss(nn.Module):
    def __init__(
        self,
        bn_mean,
        bn_var,
        sim_loss_weight=1.0,
        cov_loss_weight=1.0,
        var_loss_weight=1.0,
    ):
        super(DomainAdaptationLoss, self).__init__()
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.sim_loss_weight = sim_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.var_loss_weight = var_loss_weight

    def forward(self, y, y_prev_bn):
        # Invariance loss (matching BN statistics)
        sim_loss = 0.0
        for id in range(len(y_prev_bn)):
            sim_loss = sim_loss + self.compute_invariance_loss(
                y_prev_bn[id], self.bn_mean[id], self.bn_var[id]
            )  # Minimize the KL divergence between the batch and the running stats!
        sim_loss = self.sim_loss_weight * sim_loss
        # Covariance loss (information-maximization loss)
        cov_loss = self.cov_loss_weight * self.compute_covariance_loss(
            y
        )  # Minimize the entropy of the time step -> one-hot vectors!
        # Variance loss
        var_loss = self.var_loss_weight * (
            -1 * self.compute_variance_loss(y)
        )  # Maximize the entropy of the avg. frame -> variance within the batch!
        # Total loss
        loss = sim_loss + cov_loss + var_loss
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "cov_loss": cov_loss,
            "var_loss": var_loss,
        }

    @staticmethod
    def compute_invariance_loss(x, bn_mean, bn_var):
        # Source-free Domain Adaptation via Distributional Alignment by Matching Batch Normalization Statistics

        # x: [b, c, h, w] -> 4D output of a 2D convolutional layer

        # Compute the mean and variance of the current batch
        # We want those two to equal the running mean and running var stats tracked by the batch normalization layer
        # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py

        x_mean = x.mean(dim=[0, 2, 3])
        x_var = x.var(dim=[0, 2, 3])
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        loss = (
            torch.log(torch.sqrt(bn_var) / torch.sqrt(x_var))
            + ((x_var + (x_mean - bn_mean).pow(2)) / (2 * bn_var))
            - 0.5
        )
        loss = loss.mean()

        return loss

    @staticmethod
    def entropy(x, eps=1e-4):
        # x: [batch, features] -> logits
        x = x.softmax(dim=1)
        x = torch.clamp(x, min=eps, max=1.0)  # Avoid zero probabilities
        h = -1 * ((x * torch.log(x)).sum(dim=1))
        h = h.mean()
        return h

    def compute_covariance_loss(self, x):
        # NOTE: This is the same as the entropy regularization presented in
        # ``Domain Adaptation via Mutual Information Maximization for Handwriting Recognition''

        # x: [batch, frames, features]
        B, T, F = x.shape
        x = x.reshape(B * T, F)  # Take each frame as a sample!
        loss = self.entropy(x)
        return loss

    def compute_variance_loss(self, x):
        # x: [batch, frames, features]
        x = x.mean(dim=0)  # [frames, features]
        loss = self.entropy(x)
        return loss
