import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError
import wandb
from src.metrics.abstract_metrics import CrossEntropyMetric 

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        # Epoch-level metrics (accumulated across entire epoch)
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        # Batch-level metrics (for per-step logging)
        self.batch_node_loss = CrossEntropyMetric()
        self.batch_edge_loss = CrossEntropyMetric()
        self.batch_y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        # Update epoch-level metrics
        if flat_true_X.numel() > 0:
            self.node_loss(flat_pred_X, flat_true_X)
        if flat_true_E.numel() > 0:
            self.edge_loss(flat_pred_E, flat_true_E)
        if true_y.numel() > 0:
            self.y_loss(pred_y, true_y)
        
        # Compute differentiable losses directly for backpropagation.
        if flat_true_X.numel() > 0:
            target_X = torch.argmax(flat_true_X, dim=-1)
            loss_X = F.cross_entropy(flat_pred_X, target_X, reduction="mean")
        else:
            loss_X = masked_pred_X.sum() * 0.0

        if flat_true_E.numel() > 0:
            target_E = torch.argmax(flat_true_E, dim=-1)
            loss_E = F.cross_entropy(flat_pred_E, target_E, reduction="mean")
        else:
            loss_E = masked_pred_E.sum() * 0.0

        if true_y.numel() > 0:
            if true_y.dim() > 1:
                target_y = torch.argmax(true_y, dim=-1)
            else:
                target_y = true_y
            loss_y = F.cross_entropy(pred_y, target_y, reduction="mean")
        else:
            loss_y = pred_y.sum() * 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": loss_X,
                      "train_loss/E_CE": loss_E,
                      "train_loss/y_CE": loss_y}
            if wandb.run:
                wandb.log(to_log, commit=True)

            # Only reset batch-level metrics for per-batch logging
            self.batch_node_loss.reset()
            self.batch_edge_loss.reset()
            self.batch_y_loss.reset()
            
        return self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E + self.lambda_train[2] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss, self.batch_node_loss, self.batch_edge_loss, self.batch_y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



