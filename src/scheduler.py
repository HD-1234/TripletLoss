import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
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

    def train_one_epoch(self, lr_scheduler: LRScheduler, max_norm: float = 0.1) -> Tensor:
        """
        Trains the model for one epoch.

        Args:
            lr_scheduler (LRScheduler): The learn rate scheduler.
            max_norm (float): The maximum norm of the gradients.

        Returns:
            tensor: The average loss.
        """
        # Set to train mode
        self.model.train()

        train_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for batch in self.training_loader:
            # Unpack the batch
            data = batch["data"]
            labels = batch["labels"]

            # Get triplets from batch
            anchor, positive, negative = data[:, 0], data[:, 1], data[:, 2]
            anchor_label, positive_label, negative_label = labels[:, 0], labels[:, 1], labels[:, 2]

            # Data to device
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
            anchor_label, positive_label, negative_label = anchor_label.to(self.device), positive_label.to(self.device), negative_label.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Make predictions for this batch
            anchor, positive, negative = self.model(anchor, positive, negative, fuzzy_positional_encoding=True)

            # Compute the loss and its gradients
            loss = self.loss_fn(anchor, positive, negative, anchor_label, positive_label, negative_label)
            loss.backward()

            # Clip the gradient norm
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            # Update the model parameters by applying the computed gradients
            self.optimizer.step()

            # Add the loss for the current batch to the total training loss
            train_loss += loss.item()

        # Adjust the learning rate
        lr_scheduler.step()

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
                # Unpack the batch
                data = batch["data"]
                labels = batch["labels"]

                # Get triplets from batch
                anchor, positive, negative = data[:, 0], data[:, 1], data[:, 2]
                anchor_label, positive_label, negative_label = labels[:, 0], labels[:, 1], labels[:, 2]

                # Data to device
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor_label, positive_label, negative_label = anchor_label.to(self.device), positive_label.to(
                    self.device), negative_label.to(self.device)

                # Make predictions for this batch
                anchor, positive, negative = self.model(anchor, positive, negative)

                # Compute the loss for the current batch and add it to the total validation loss
                val_loss += self.loss_fn(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    anchor_label=anchor_label,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    eval_mode=True
                )

        return val_loss
