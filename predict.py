import argparse
import os
import shutil
from collections import defaultdict

import torch

from pathlib import Path
from torch import nn

import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import InferenceDataset
from src.modelloader import ModelLoader
from src.utils import set_seed, write_log_message, set_worker_seed


def predict(model: nn.Module, data_loader: DataLoader, device: str, temp_folder: str) -> dict:
    """
    Predicts the embeddings for each element in the dataloader.

    Args:
        model (nn.Module): The model to use for inference.
        data_loader (DataLoader): The data loader for the corresponding dataset.
        device (str): The device to use.
        temp_folder (str): Path to the temporary folder.

    Returns:
        dict: A dict containing the original file path as key and the path to the corresponding tensor as value.
    """
    # Set the model to eval mode
    model.eval()

    embeddings = defaultdict(str)
    with torch.no_grad():
        for batch in data_loader:
            # Unpack the batch
            image_tensors, paths = batch

            # Data to device
            image_tensors = image_tensors.to(device)

            # Make predictions for this batch
            prediction = model(image_tensors, image_tensors, image_tensors)[0]

            for ind, path in enumerate(paths):
                file_name = os.path.basename(path)
                base_name, _ = os.path.splitext(file_name)
                embedding_path = os.path.join(temp_folder, f"{base_name}.pt")

                # Save the embedding
                torch.save(obj=prediction[ind], f=embedding_path)

                # Store the path to the embedding
                embeddings[path] = embedding_path

    return embeddings


def calculate_distance(emb_1: torch.Tensor, emb_2: torch.Tensor) -> float:
    """
    Calculates the distance between two embeddings.

    Args:
        emb_1 (torch.Tensor): The first embedding.
        emb_2 (torch.Tensor): The second embedding.

    Returns:
        float: The distance between the two embeddings.
    """
    # Normalize the embeddings
    emb_1 = F.normalize(emb_1, p=2, dim=0)
    emb_2 = F.normalize(emb_2, p=2, dim=0)

    # Calculate the similarity between the two embeddings
    score = torch.matmul(emb_1, torch.t(emb_2)).item()

    # Calculate the distance
    distance = 1.0 - score

    return distance


def cluster_files(embeddings: dict, threshold: float, strategy: str = "avg") -> dict:
    """
    Clusters files based on their embeddings.

    Args:
        embeddings (dict): A dictionary mapping file paths to the paths of their corresponding embeddings.
        threshold (float): The threshold for clustering files.
        strategy (str): The clustering strategy.

    Returns:
        dict: A dictionary mapping cluster IDs to lists of file paths.
    """
    clusters = defaultdict(list)

    for file, emb in embeddings.items():
        # Load the embedding for the current file
        emb = torch.load(emb, weights_only=True)

        target_cluster = str(len(clusters))
        for cluster, clustered_files in clusters.items():
            # Check if every distance is below the threshold
            if strategy == "all":
                for cf in clustered_files:
                    # Load the corresponding embedding
                    cluster_emb = embeddings[cf]
                    cluster_emb = torch.load(cluster_emb, weights_only=True)

                    # Calculate the distance between the embeddings
                    distance = calculate_distance(emb_1=emb, emb_2=cluster_emb)

                    # If any distance is higher than the threshold, iterate over the next cluster
                    if distance > threshold:
                        cluster = target_cluster
                        break

                # If every distance is lower than or equal to the threshold, set the target cluster to the current
                # cluster
                target_cluster = cluster

            # The average distance has to be lower than the threshold
            else:
                total_distance = 0.0
                for cf in clustered_files:
                    # Load the corresponding embedding
                    cluster_emb = embeddings[cf]
                    cluster_emb = torch.load(cluster_emb, weights_only=True)

                    # Add the calculated distance to the total distance
                    total_distance += calculate_distance(emb_1=emb, emb_2=cluster_emb)

                # Calculate the average distance and compare it with the threshold value.
                avg_distance = total_distance / len(clustered_files)
                if avg_distance <= threshold:
                    target_cluster = cluster
                    break

        # Append the file to the specific cluster
        clusters[target_cluster].append(file)

    return clusters


def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-path', 
        type=Path, 
        required=True, 
        help='Path to the input directory containing files to be sorted.'
    )
    parser.add_argument(
        '-o', '--output-path', 
        type=Path, 
        required=True, 
        help='Path to the output directory where sorted files will be stored.'
    )
    parser.add_argument(
        '-m', '--model', 
        type=Path, 
        required=True, 
        help='Path to the model.'
    )
    parser.add_argument(
        '-t', '--threshold', 
        type=float, 
        required=True, 
        help='Threshold for clustering files based on their embeddings.'
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
            'ViT_L_32', 'ViTAR_B_16', 'ViTAR_L_16'
        ]
    )
    parser.add_argument(
        '--clustering-strategy', 
        type=str, 
        default='avg', 
        choices=['all', 'avg'], 
        help='Choose the clustering strategy (Options: "all", "avg"). "all" means that every distance  between a file '
             'and the files in an existing cluster must be lower than or equal to the threshold. "avg" means that the '
             'average distance between a file and the files in an existing cluster must be lower than or equal to the '
             'threshold.'
    )
    args = parser.parse_args()

    # Set gpu, mps or cpu
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    write_log_message(f"Using {device} device.")

    # Check if we got a path to a folder
    if not args.input_path.is_dir():
        raise ValueError(f"The path '{args.input_path}' is not a folder.")

    # Create output folder
    if os.path.exists(args.output_path):
        raise ValueError(f"The path '{args.output_path}' already exists.")
    else:
        os.mkdir(args.output_path)

    # Sets the seeds
    generator = set_seed(seed=args.seed)

    # Build the test data loader
    test_set = InferenceDataset(path=args.input_path, img_size=args.image_size)
    test_loader = DataLoader(
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

    # Ensure the temporary output folder does not exist
    temp_folder = os.path.join(args.output_path, "temp")
    if os.path.exists(temp_folder):
        raise Exception(f"Path '{temp_folder}' already exists.")

    # Create the temporary output folder
    os.mkdir(temp_folder)

    # Predict embeddings
    write_log_message("Predicting embeddings.")
    embeddings = predict(
        model=model,
        data_loader=test_loader,
        device=device,
        temp_folder=temp_folder
    )

    # Sort embeddings
    embeddings = dict(sorted(embeddings.items(), key=lambda x: x[1]))

    # Cluster files based on the threshold and the chosen clustering strategy
    write_log_message(f"Clustering files with the strategy set to: '{args.clustering_strategy}'.")
    clustered_files = cluster_files(embeddings=embeddings, threshold=args.threshold, strategy=args.clustering_strategy)

    # Save the clustered files
    write_log_message("Saving the clustered files.")
    for i, paths in clustered_files.items():
        cluster_path = os.path.join(args.output_path, f"cluster_{i}")

        # If the current cluster contains only a single file, ignore the cluster
        if len(paths) == 1:
            cluster_path = os.path.join(args.output_path, "no_cluster")

        # Create a folder for new clusters
        if not os.path.exists(cluster_path):
            os.mkdir(cluster_path)

        # Iterate over each path and copy the corresponding file to the relevant cluster
        for path in paths:
            file_name = os.path.basename(path)
            shutil.copy(src=path, dst=os.path.join(cluster_path, file_name))

    # Clean up temp files
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)


if __name__ == "__main__":
    run_inference()
