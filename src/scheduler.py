import random

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm


class Scheduler:
    def __init__(
            self,
            model: nn.Module,
            device: str,
            img_size: int,
            loss_fn: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler: LRScheduler,
            training_loader: DataLoader,
            validation_loader: DataLoader,
            max_norm: float = 0.1,
            max_scale_factor: int = 1,
            avg_loss: bool = False
    ) -> None:
        """
        Initializes the scheduler.

        Args:
            model (nn.Module): The model to train.
            device (str): The device to use for training.
            img_size (int): The image size.
            loss_fn (nn.Module): The loss function to use.
            optimizer (optim.Optimizer): The optimizer to use.
            lr_scheduler (LRScheduler): The learn rate scheduler.
            training_loader (DataLoader): The training data loader.
            validation_loader (DataLoader): The validation data loader.
            max_norm (float): The maximum norm of the gradients.
            max_scale_factor (int): The maximum scale factor for random resizing.
            avg_loss (bool): Whether to report the average loss or the running loss.
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_norm = max_norm
        self.max_scale_factor = max_scale_factor

        # Validate the parameters
        if max_scale_factor < 1:
            raise ValueError("'max_scale_factor' must be greater or equal to 1.")
        if max_norm < 0.0:
            raise ValueError("'max_norm' must be greater or equal to 0.")

        # Train/Val loader
        self.training_loader = training_loader
        self.validation_loader = validation_loader

        # Transformations
        self.augmentation_transformation = transforms.Compose(
            [
                transforms.RandomRotation(
                    degrees=(-2, 2)
                ),
                transforms.ColorJitter(
                    brightness=(0.85, 1.0),
                    contrast=(0.85, 1.0),
                    saturation=(0.85, 1.0),
                    hue=(-0.05, 0.05)
                ),
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(1.5, 2.0)
                ),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=2,
                    p=0.5
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.1,
                    p=1.0
                )
            ]
        )
        self.normalization = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        # Report the average loss or the running loss
        self.avg_loss = avg_loss

    def _get_scale_factor(self) -> int:
        """
        Get a random scale factor for resizing.

        Returns:
            int: The random scale factor.
        """
        scale_factor = 1
        if self.max_scale_factor != 1 and random.randint(0, 1) == 1:
            scale_factor = random.randint(2, self.max_scale_factor)
        return scale_factor

    def _apply_transformation(
            self,
            input_tensor: Tensor,
            augmentation: bool = False,
            random_scaling: bool = False,
            scale_factor: int = 1
    ) -> Tensor:
        """
        Applies transformations to the input tensor.

        Args:
            input_tensor (Tensor): The input tensor.
            augmentation (bool): Whether to apply data augmentation or not.
            random_scaling (bool): Whether to apply a random scale factor or not.
            scale_factor (int): The random scale factor.

        Returns:
            Tensor: The transformed and normalized tensor.
        """
        if random_scaling and self.max_scale_factor != 1:
            # Get a random scale factor
            scale_factor = self._get_scale_factor()

        if augmentation:
            # Apply random transformations
            input_tensor = self.augmentation_transformation(input_tensor)

        # Resize the image
        transform = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size * scale_factor, self.img_size * scale_factor))
            ]
        )
        input_tensor = transform(input_tensor)

        return self.normalization(input_tensor)

    def train_one_epoch(self, augmentation: bool) -> Tensor:
        """
        Trains the model for one epoch.

        Args:
            augmentation (bool): Whether to apply data augmentation or not.

        Returns:
            Tensor: The average loss.
        """
        # Set the model to train mode
        self.model.train()

        # Initialize the total training loss
        train_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Training loop
        for batch in tqdm(self.training_loader):
            # Get triplets and labels from batch
            anchor, positive, negative = batch["anchors"], batch["positives"], batch["negatives"]
            anchor_label, positive_label, negative_label = (
                batch["anchor_labels"],
                batch["positive_labels"],
                batch["negative_labels"]
            )

            # Get the batch size
            batch_size = anchor.shape[0]

            # Move data to device
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
            anchor_label, positive_label, negative_label = (
                anchor_label.to(self.device), positive_label.to(self.device), negative_label.to(self.device)
            )

            # Apply the transformations
            anchor, positive, negative = (
                self._apply_transformation(anchor, augmentation=augmentation, random_scaling=True),
                self._apply_transformation(positive, augmentation=augmentation, random_scaling=True),
                self._apply_transformation(negative, augmentation=augmentation, random_scaling=True)
            )

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass through the model
            anchor, positive, negative = self.model(anchor, positive, negative, fuzzy_positional_encoding=True)

            # Compute the loss and its gradients
            loss = self.loss_fn(anchor, positive, negative, anchor_label, positive_label, negative_label)
            loss.backward()

            # Clip the gradients
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            # Update the model parameters by applying the computed gradients
            self.optimizer.step()

            # Add the average loss or the running loss for the current batch
            train_loss += loss.item()
            if self.avg_loss:
                train_loss *= batch_size

        # Adjust the learning rate
        self.lr_scheduler.step()

        if self.avg_loss:
            # Calculate the average loss
            train_loss /= len(list(self.training_loader.sampler))

        return train_loss

    def evaluate(self) -> Tensor:
        """
        Evaluates the model.

        Returns:
            Tensor: The average validation loss.
        """
        # Set the model to eval mode
        self.model.eval()

        # Initialize the total validation loss
        val_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device) / len(self.validation_loader)

        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(self.validation_loader):
                # Get triplets and labels from the batch
                anchor, positive, negative = batch["anchors"], batch["positives"], batch["negatives"]
                anchor_label, positive_label, negative_label = (
                    batch["anchor_labels"],
                    batch["positive_labels"],
                    batch["negative_labels"]
                )

                # Get the batch size
                batch_size = anchor.shape[0]

                # Apply the transformations
                anchor, positive, negative = (
                    self._apply_transformation(anchor),
                    self._apply_transformation(positive),
                    self._apply_transformation(negative)
                )

                # Move data to device
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor_label, positive_label, negative_label = (
                    anchor_label.to(self.device), positive_label.to(self.device), negative_label.to(self.device)
                )

                # Forward pass through the model
                anchor, positive, negative = self.model(anchor, positive, negative)

                # Compute the average loss or the running loss for the current batch
                val_loss += self.loss_fn(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    anchor_label=anchor_label,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    eval_mode=True
                )
                if self.avg_loss:
                    val_loss *= batch_size

        if self.avg_loss:
            # Calculate the average loss
            val_loss /= len(list(self.validation_loader.sampler))

        return val_loss
