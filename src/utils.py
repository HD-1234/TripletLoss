import os
import random
from pathlib import Path
from typing import Dict, Tuple

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


def validate(model, data_loader, threshold, device) -> Tuple[float, float, float, float]:
    """
    Evaluates the model and returns precision, recall, f1-score and accuracy.

    Returns:
        tuple: a Tuple of floats.
    """
    # Set to eval mode
    model.eval()

    # Initialize tensors to accumulate results
    batch_count = len(data_loader)
    tp = torch.zeros(batch_count, device=device)
    fp = torch.zeros(batch_count, device=device)
    fn = torch.zeros(batch_count, device=device)
    tn = torch.zeros(batch_count, device=device)

    with torch.no_grad():
        for ind, batch in enumerate(data_loader):
            # Get triplets from batch
            anchor, positive, negative = batch[:, 0], batch[:, 1], batch[:, 2]

            # Data to device
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Make predictions for this batch
            out_anchor, out_positive, out_negative = model(anchor, positive, negative)

            # Calculate L2 norm (Euclidean distance) between anchors and positive/negative samples
            dist_ap = torch.norm(out_anchor - out_positive, dim=1)
            dist_an = torch.norm(out_anchor - out_negative, dim=1)

            # Get true positives and false positives
            positive_criteria = (dist_ap <= threshold)
            tp_c = torch.sum(positive_criteria).item()
            fp_c = torch.sum(~positive_criteria).item()

            # Get true negatives and false negatives
            negative_criteria = (dist_an > threshold)
            fn_c = torch.sum(~negative_criteria).item()
            tn_c = torch.sum(negative_criteria).item()

            # Update tensors
            tp[ind] = tp_c
            fp[ind] = fp_c
            fn[ind] = fn_c
            tn[ind] = tn_c

    # Sum up results
    tp = torch.sum(tp).item()
    fp = torch.sum(fp).item()
    fn = torch.sum(fn).item()
    tn = torch.sum(tn).item()

    # Calculate precision, recall, f1 and accuracy
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0.0
    recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return precision, recall, f1_score, accuracy


def write_log_message(*args) -> None:
    """
    Writes a log message to the console.

    Args:
        *args: Variable number of arguments to write a log message.
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
