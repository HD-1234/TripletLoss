import argparse
import numpy as np
from pathlib import Path
import torch

from src.dataloader import TripletDataset
from src.modelloader import ModelLoader
from src.utils import set_seed, write_log_message, set_worker_seed, validate


def val():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=Path, required=True)
    parser.add_argument('-m', '--model', type=Path, required=True)
    parser.add_argument('-s', '--image-size', type=int, default=224, help='Set the image size.')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Set the batch size.')
    parser.add_argument('-n', '--num-workers', type=int, default=2, help='Set the number of workers.')
    parser.add_argument('-t', '--threshold', type=float, default=2.0, help='Set the threshold')
    parser.add_argument('--seed', type=int, default=42, help='Set the seed.')
    parser.add_argument('--model-name', type=str, default='ResNeXt50', choices=['ResNeXt50'], 
                        help='Choose the model to use.')
    args = parser.parse_args()

    # Set gpu, mps or cpu
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    write_log_message(f"Using {device} device.")

    # Sets the seeds
    generator = set_seed(seed=args.seed)

    # Build the test data loader
    test_set = TripletDataset(path=args.path, size=args.image_size, augmentation=False)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Define your model
    model_loader = ModelLoader(model_name=args.model_name)
    model = model_loader.load_model()
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            args.model,
            weights_only=False,
            map_location=torch.device(device=device)
        )
    )

    # Evaluate
    prec, rec, f1, acc = validate(
        model=model,
        data_loader=test_loader,
        threshold=args.threshold,
        device=device
    )

    # Print metrics
    print(f"Precision: {prec:.2f}.")
    print(f"Recall: {rec:.2f}.")
    print(f"F1-Score: {f1:.2f}.")
    print(f"Accuracy: {acc:.2f}.")


if __name__ == "__main__":
    val()
