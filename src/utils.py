import os
import random
from pathlib import Path
from typing import Dict

import yaml
from datetime import datetime

import numpy as np
import torch


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


def write_log_message(*args) -> None:
    """
    Writes a log message to the console.

    Args:
        *args: Variable number of arguments to write into a log message.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = " ".join(str(a) for a in args)
    print(f"[{timestamp}] {message}")


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
