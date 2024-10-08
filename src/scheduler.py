import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as optim


class Scheduler:
    def __init__(self, model: nn.Module, device: str, loss_fn: nn.Module, optimizer: optim.Optimizer,
                 training_loader: DataLoader, validation_loader: DataLoader) -> None:
        """
        Initializes the scheduler.

        Args:
            model (nn.Module): The model to train.
            device (str): The device to use for training.
            loss_fn (nn.Module): The loss function to use.
            optimizer (optim.Optimizer): The optimizer to use.
            training_loader (DataLoader): The training data loader.
            validation_loader (DataLoader): The validation data loader.
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Train/Val loader
        self.training_loader = training_loader
        self.validation_loader = validation_loader

    @staticmethod
    def _semi_hard_mining(anchors: Tensor, positives: Tensor, negatives: Tensor, margin: float = 1.0) -> Tensor:
        """
        Performs semi-hard mining.

        Args:
            anchors (Tensor): The anchor samples.
            positives (Tensor): The positive samples.
            negatives (Tensor): The negative samples.

        Returns:
            Tensor: The semi hard negative samples.
        """
        # Calculate pairwise distance between anchors and positive/negative samples
        dist_ap = torch.cdist(anchors, positives)
        dist_an = torch.cdist(anchors, negatives)

        # Define Condition
        semi_hard_negative_mask = (dist_an < dist_ap + margin) & (dist_an > dist_ap)

        # Get indices of the minimum semi-hard negative samples, setting unmatched distances to infinity
        indices = torch.where(semi_hard_negative_mask, dist_an, torch.inf).min(dim=1).indices

        return negatives[indices, :]

    def train_one_epoch(self) -> Tensor:
        """
        Trains the model for one epoch.

        Returns:
            tensor: The average loss.
        """
        # Set to train mode
        self.model.train()

        train_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for batch in self.training_loader:
            # Get triplets from batch
            anchor, positive, negative = batch[:, 0], batch[:, 1], batch[:, 2]

            # Data to device
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Make predictions for this batch
            out_anchor, out_positive, out_negative = self.model(anchor, positive, negative)

            # Perform semi-hard mining
            semi_hard_negative = self._semi_hard_mining(out_anchor, out_positive, out_negative)

            # Compute the loss and its gradients
            loss = self.loss_fn(out_anchor, out_positive, semi_hard_negative)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Add loss to total loss
            train_loss += loss.item()

        return train_loss

    def evaluate(self) -> Tensor:
        """
        Evaluates the model.

        Returns:
            tensor: The average validation loss.
        """
        # Set to eval mode
        self.model.eval()

        val_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for batch in self.validation_loader:
                # Get triplets from batch
                anchor, positive, negative = batch[:, 0], batch[:, 1], batch[:, 2]

                # Data to device
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                # Make predictions for this batch
                out_anchor, out_positive, out_negative = self.model(anchor, positive, negative)

                # Compute the loss and add it to total loss
                val_loss += self.loss_fn(out_anchor, out_positive, out_negative)

        return val_loss
