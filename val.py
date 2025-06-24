import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from src.dataset import TripletDataset
from src.modelloader import ModelLoader
from src.utils import set_seed, write_log_message, set_worker_seed


def calculate_distances(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the distances between anchors and positives and anchors and negatives.

    Args:
        model (nn.Module): The model to use for inference.
        data_loader (DataLoader): The data loader for the corresponding dataset.
        device (str): The device to use.

    Returns:
        Tuple: A tuple containing the distances between anchors and positives and anchors and negatives.
    """
    # Set to eval mode
    model.eval()

    # Initialize tensors to accumulate results
    num_samples = len(data_loader.dataset)
    positives = torch.zeros(num_samples, device=device)
    negatives = torch.zeros(num_samples, device=device)

    with torch.no_grad():
        for ind, batch in enumerate(data_loader):
            # Unpack the batch
            data, _ = batch

            # Get triplets from batch
            anchor, positive, negative = data[:, 0], data[:, 1], data[:, 2]

            # Standard batch size
            batch_size = data_loader.batch_size

            # Calculate the start and end indices for the current batch
            start_index = ind * batch_size
            end_index = start_index + anchor.size(0)

            # Data to device
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Make predictions for this batch
            anchor, positive, negative = model(anchor, positive, negative)

            # Normalize the embeddings
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

            # Calculate the similarity between anchors and positives
            positive_scores = torch.matmul(anchor, torch.t(positive))

            # Calculate the distance for positives
            positive_dist = torch.ones(anchor.size(0), device=device) - positive_scores

            # Update positives
            positives[start_index:end_index] = torch.diagonal(positive_dist, offset=0)

            # Calculate the similarity between anchors and negatives
            negative_scores = torch.matmul(anchor, torch.t(negative))

            # Calculate the distance for negatives
            negative_dist = torch.ones(anchor.size(0), device=device) - negative_scores

            # Update negatives
            negatives[start_index:end_index] = torch.diagonal(negative_dist, offset=0)

    return positives, negatives


def calculate_metrics(
        positives: torch.tensor,
        negatives: torch.tensor,
        eps: float = 1e-12,
        threshold: Optional[float] = None
) -> Tuple[float, float, float, float, float]:
    """
    Calculates accuracy, F1-score, recall and precision for different thresholds.

    Args:
        positives (torch.Tensor): Positive distances.
        negatives (torch.Tensor): Negative distances.
        eps (float): Small epsilon to avoid division by zero.
        threshold (Optional[float]): The threshold to use for calculating the metrics. If not provided, the best
            threshold will be calculated.

    Returns:
        Tuple: A tuple containing the best threshold and the corresponding accuracy, F1-score, recall, and precision.
    """
    # Calculate the minimum and maximum values for both tensors
    min_pos = torch.min(positives)
    max_pos = torch.max(positives)
    min_neg = torch.min(negatives)
    max_neg = torch.max(negatives)

    # No overlap between positive and negative distances
    if max_pos <= min_neg:
        threshold = (max_pos + min_neg) / 2
        accuracy, f1_score, recall, precision = [1.0, 1.0, 1.0, 1.0]

    # If the minimum positive value is greater than or equal to the maximum negative value, set all metrics to 0
    elif min_pos >= max_neg:
        threshold, accuracy, f1_score, recall, precision = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Calculate the best threshold
    else:
        if threshold is None:
            # Generate the thresholds between the minimum positive and maximum negative distances
            thresholds = torch.linspace(min_pos, max_neg, 1000)
        else:
            thresholds = torch.tensor([threshold])

        results = torch.zeros((thresholds.shape[0], 4))

        # Iterate over each threshold
        for ind, threshold in enumerate(thresholds):
            # Calculate true positives and false negatives
            tp = torch.sum(positives < threshold)
            fn = positives.size(0) - tp

            # Calculate true negatives and false positives
            tn = torch.sum(negatives >= threshold)
            fp = negatives.size(0) - tn

            # Calculate metrics
            accuracy = (tp + tn) / (positives.size(0) + negatives.size(0) + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1_score = 2 * (precision * recall) / (precision + recall + eps)

            # Store the results
            results[ind, 0] = accuracy
            results[ind, 1] = f1_score
            results[ind, 2] = recall
            results[ind, 3] = precision

        # Find the best index
        scores = torch.zeros(results[:, 0].shape)
        scores += ((results[:, 3] * 0.9) + (results[:, 1] * 0.1))
        index = torch.argmax(scores[:])

        # Get the best threshold and the corresponding metrics
        threshold = thresholds[index].item()
        accuracy = results[index, 0].item()
        f1_score = results[index, 1].item()
        recall = results[index, 2].item()
        precision = results[index, 3].item()

    return threshold, accuracy, f1_score, recall, precision


def val():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path', 
        type=Path, 
        required=True, 
        help='Path to the test dataset directory.'
    )
    parser.add_argument(
        '-m', '--model', 
        type=Path, 
        required=True, 
        help='Path to the model.'
    )
    parser.add_argument(
        '-s', '--image-size', 
        type=int, 
        default=224, 
        help='Size of the input images.'
    )
    parser.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=16, 
        help='Batch size for evaluation.'
    )
    parser.add_argument(
        '-n', '--num-workers', 
        type=int, 
        default=4, 
        help='Number of workers for the dataloader.'
    )
    parser.add_argument(
        '-t', '--threshold', 
        type=float, 
        default=None, 
        help='The threshold for calculating the metrics. If not provided, the best threshold will be calculated.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='ResNeXt50', 
        help='Model architecture to use.', 
        choices=[
            'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50', 'ResNeXt101', 'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 
            'ViT_L_32', 'ViTAR_B_16'
        ]
    )
    args = parser.parse_args()

    # Set gpu, mps or cpu
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    write_log_message(f"Using {device} device.")

    # Sets the seeds
    generator = set_seed(seed=args.seed)

    # Build the test data loader
    test_set = TripletDataset(path=args.path, img_size=args.image_size, augmentation=False)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Load the model
    model_loader = ModelLoader(model_name=args.model_name, image_size=args.image_size)
    model = model_loader.load_model()
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            args.model,
            weights_only=False,
            map_location=torch.device(device=device)
        )
    )

    # Evaluate the model
    positives, negatives = calculate_distances(model=model, data_loader=test_loader, device=device)

    # Calculate the evaluation metrics
    threshold, accuracy, f1_score, recall, precision = calculate_metrics(
        positives=positives,
        negatives=negatives,
        threshold=args.threshold
    )

    # Print the evaluation metrics
    write_log_message(f"Results at{' best' if not args.threshold else ''} threshold of '{threshold:.2f}'")
    write_log_message(f"Accuracy: {accuracy:.2f}")
    write_log_message(f"F1-Score: {f1_score:.2f}")
    write_log_message(f"Recall: {recall:.2f}")
    write_log_message(f"Precision: {precision:.2f}")


if __name__ == "__main__":
    val()
