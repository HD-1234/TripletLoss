import os
import random
from pathlib import Path
from typing import Dict, Any

import yaml

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

import logging

logging.basicConfig(format='[%(asctime)s - %(levelname)s] %(message)s', level=logging.INFO, datefmt='%H:%M:%S')


class BestResult:
    def __init__(self, epoch: int = 0, loss: float = float('inf')) -> None:
        self.epoch = epoch
        self.loss = loss


def set_seed(seed: int, deterministic_algorithms: bool = False) -> torch.Generator:
    """
    Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
        deterministic_algorithms (bool): Whether to use deterministic algorithms or not.

    Returns:
        torch.Generator: A PyTorch Generator object with the given seed.
    """
    if deterministic_algorithms:
        # Set deterministic mode for CUDA, MPS and CPU backends
        torch.backends.cudnn.deterministic = True
        torch.backends.mps.deterministic = True
        torch.backends.cpu.deterministic = True

    # Create a PyTorch Generator object on the CPU device
    generator = torch.Generator(device='cpu')

    # Set the seed for the generator
    generator.manual_seed(seed)

    # Set the global PyTorch seed
    torch.manual_seed(seed)

    # Set the seed for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    return generator


def set_worker_seed(worker_id):
    """
    Worker initialization function that sets the seed for each worker.

    Source:
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_lr_factor(
        lr0: float,
        lr1: float,
        epoch: int,
        max_epochs: int,
        schedule_type: str = 'fixed',
        steps: list = None
) -> float:
    """
    Computes the learning rate factor for a given epoch based on the specified schedule type.

    Args:
        lr0 (float): The initial learning rate.
        lr1 (float): The target learning rate at the end of the training.
        epoch (int): The current epoch.
        max_epochs (int): The total number of epochs.
        schedule_type (str): The type of learning rate schedule ('fixed', 'linear', 'steps' or 'exponential').
        steps (list): The epoch in which the learning rate should be adjusted.

    Returns:
        float: The learning rate factor to be applied to the initial learning rate.
    """
    # Set steps & target learning rate
    steps = [] if steps is None else list(dict.fromkeys(steps))
    lr1 = lr1 if lr1 else lr0

    # If the current epoch is 0 or the initial learning rate is already the target learning rate or the type is set to
    # 'fixed', no adjustment is needed.
    if epoch == 0 or lr0 == lr1 or schedule_type == 'fixed':
        return 1.0

    # If the current epoch is the last epoch, return the factor that will adjust the learning rate to the target
    # learning rate.
    if epoch >= max_epochs - 1:
        return lr1 / lr0

    # Exponential learning rate schedule
    if schedule_type == 'exponential':
        # Add a small epsilon to avoid division by zero
        eps = 1e-12

        # Calculate the exponential decay factor
        gamma = (lr1 / (lr0 + eps)) ** (1 / (max_epochs - 1))

        # Compute the learning rate factor for the current epoch
        factor = gamma ** epoch

    # Step based adjustments
    elif schedule_type == 'steps':
        factor = 1.0
        for step in steps[::-1]:
            if epoch + 1 >= step:
                # Get index
                ind = steps.index(step)

                # Calculate the difference between the target learning rate and the initial learning rate
                delta = lr1 - lr0

                # Calculate the step size for each epoch
                step_size = delta / (len(steps))

                # Compute the learning rate for the current epoch
                lr = lr0 + (step_size * (ind + 1))

                # Compute the learning rate factor for the current epoch
                factor = lr / lr0
                break

    # Linear learning rate schedule
    else:
        # Calculate the difference between the target learning rate and the initial learning rate
        delta = lr1 - lr0

        # Calculate the step size for each epoch
        step_size = delta / (max_epochs - 1)

        # Compute the learning rate for the current epoch
        lr = lr0 + step_size * epoch

        # Compute the learning rate factor for the current epoch
        factor = lr / lr0

    return factor


def write_log_message(message: str) -> None:
    """
    Writes a log message to the console.

    Args:
        message (str): The message to print.
    """
    logging.info(message)


def write_hyperparameters_to_yaml(path: str, hyperparameters: Dict) -> None:
    """
    Saves hyperparameters to a YAML file.

    Args:
        path (str): Directory where the YAML file will be saved.
        hyperparameters (Dict): Hyperparameters to save.
    """
    # Convert any path object to string
    hyperparameters = {k: str(v) if isinstance(v, Path) else v for k, v in hyperparameters.items()}

    # Save hyperparameters to yaml
    yaml_file_path = os.path.join(path, 'hyperparameters.yaml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(hyperparameters, file, default_flow_style=False, sort_keys=False)


def save_training_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
):
    """
    Saves the current checkpoint to allow resuming rhe training.

    Args:
        path (str): The path to save the checkpoint.
        epoch (int): The current epoch.
        model (nn.Module): The model.
        optimizer (Optimizer): The optimizer.
        lr_scheduler (LRScheduler): The learning rate scheduler.
    """
    training_state = {
        "current_epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }

    # Save latest model
    torch.save(training_state, os.path.join(path, f"latest.pth"))


def load_pretrained_weights(path: str, model_state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    """
    Load and preprocess pretrained model weights.

    Args:
        path (str): The path to the pretrained weights.
        model_state_dict (Dict[str, Any]): The current model state dict.
        device (str): The device.

    Returns:
        Dict[str, torch.Tensor]: The loaded and preprocessed pretrained model weights.
    """
    write_log_message(f"Loading pretrained weights from '{path}'.")

    def construct_pretrained_weights(pretrained_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "model_state_dict" in pretrained_weights:
            pretrained_weights = pretrained_weights["model_state_dict"]

        output = dict()
        for k, v in pretrained_weights.items():
            if not k.startswith("embedding_model."):
                k = f"embedding_model.{k}"
            output[k] = v
        return output

    # Load the pretrained model weights
    pretrained_state_dict = construct_pretrained_weights(
        torch.load(path, weights_only=True, map_location=device)
    )

    # Additional keys in pretrained weights
    additional_keys_pretrained = list(set(pretrained_state_dict.keys()) - set(model_state_dict.keys()))
    if additional_keys_pretrained:
        write_log_message(
            f"Additional keys in pretrained weights (Total: {len(additional_keys_pretrained)}):"
            f"\n{additional_keys_pretrained}"
        )

    # Additional keys in model weights
    additional_keys_model = list(set(model_state_dict.keys()) - set(pretrained_state_dict.keys()))
    if additional_keys_model:
        write_log_message(
            f"Missing keys in pretrained weights (Total: {len(additional_keys_model)}):\n{additional_keys_model}"
        )

    return pretrained_state_dict
