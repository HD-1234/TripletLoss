import argparse
import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
import torch.optim as optim

from src.dataset import TripletDataset
from src.loss import SelectivelyContrastiveTripletLoss, TripletLossWithMargin
from src.scheduler import Scheduler
from src.modelloader import ModelLoader
from src.utils import (
    write_log_message,
    set_seed,
    write_hyperparameters_to_yaml,
    set_worker_seed,
    calculate_lr_factor,
    BestResult
)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-set', type=Path, required=True,
                        help='Path to the training dataset directory.')
    parser.add_argument('-v', '--val-set', type=Path, required=True,
                        help='Path to the validation dataset directory.')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Batch size for training and validation.')
    parser.add_argument('-s', '--image-size', type=int, default=224,
                        help='Size of the input images.')
    parser.add_argument('-n', '--num-workers', type=int, default=2,
                        help='Number of workers for the dataloader.')
    parser.add_argument('-m', '--margin', type=float, default=1.0,
                        help='Margin value for the triplet loss function.')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0001,
                        help='Initial learning rate for the optimizer.')
    parser.add_argument('--target-lr', type=float, default=None,
                        help='Learning rate at the end of the training. If not specified, the learning rate is fixed.')
    parser.add_argument('--lr-steps', nargs='+', type=int, default=None,
                        help='Epochs at which the learning rate should be adjusted. The new learning rate is '
                             'calculated based on the difference between the initial learning rate and the target '
                             'learning rate divided by the number of steps.')
    parser.add_argument('--lr-schedule', type=str, default='fixed',
                        choices=['fixed', 'linear', 'exponential', 'steps'],
                        help='Type of learning rate schedule (Options: "fixed", "linear", "exponential", "steps").')
    parser.add_argument('--positive-mining-strategy', type=str, default='random',
                        choices=['random', 'easy'],
                        help='Strategy for selecting positive samples (Options: "random", "easy").')
    parser.add_argument('--negative-mining-strategy', type=str, default='semi-hard',
                        choices=['random', 'semi-hard', 'hard'],
                        help='Strategy for selecting negative samples (Options: "random", "semi-hard", "hard").')
    parser.add_argument('--loss-function', type=str, default='TripletLoss', choices=['SCTLoss', 'TripletLoss'],
                        help='Loss function to use (Options: "SCTLoss", "TripletLoss").')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for the positive and negative scores. Only relevant when using SCTLoss.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--augment', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to apply data augmentation during the training.')
    parser.add_argument('--early-stopping', type=int, default=-1,
                        help='Number of epochs without improvement before stopping the training. Set to -1 to disable.')
    parser.add_argument('--pretrained-weights', type=Path, default=None,
                        help='Path to the pre-trained weights.')
    parser.add_argument('--log-folder', type=Path, default='./runs',
                        help='Directory where logs and other outputs will be saved.')
    parser.add_argument('--deterministic-algorithms', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether deterministic algorithms should be used during training.')
    parser.add_argument('--model-name', type=str, default='ResNeXt50', help='Model architecture to train.',
                        choices=['ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50', 'ResNeXt101', 'ViT_B_16',
                                 'ViT_B_32', 'ViT_L_16', 'ViT_L_32'])
    args = parser.parse_args()

    # Set gpu, mps or cpu
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    write_log_message(f"Using {device} device.")

    # Create a log folder with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_folder, timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save the hyperparameters
    write_hyperparameters_to_yaml(path=log_dir, hyperparameters=vars(args))

    # Initialize the summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Sets the seed
    generator = set_seed(seed=args.seed, deterministic_algorithms=args.deterministic_algorithms)

    # Build the training data loader
    training_set = TripletDataset(path=args.train_set, size=args.image_size, augmentation=args.augment)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Build the validation Data loader
    validation_set = TripletDataset(path=args.val_set, size=args.image_size)
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Load model to device
    model_loader = ModelLoader(
        model_name=args.model_name,
        pretrained_weights=args.pretrained_weights,
        image_size=args.image_size
    )
    model = model_loader.load_model()
    model = model.to(device)

    # Define loss function and optimizer
    if args.loss_function == "SCTLoss":
        loss_fn = SelectivelyContrastiveTripletLoss(
            positive_mining_strategy=args.positive_mining_strategy,
            negative_mining_strategy=args.negative_mining_strategy,
            temperature=args.temperature,
            margin=args.margin
        )
    else:
        loss_fn = TripletLossWithMargin(
            positive_mining_strategy=args.positive_mining_strategy,
            negative_mining_strategy=args.negative_mining_strategy,
            margin=args.margin
        )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Sets the learning rate
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_epoch: calculate_lr_factor(
            lr0=args.learning_rate,
            lr1=args.learning_rate if args.lr_schedule == 'fixed' else args.target_lr,
            epoch=current_epoch,
            max_epochs=args.epochs,
            schedule_type=args.lr_schedule,
            steps=args.lr_steps
        )
    )

    # Init scheduler
    scheduler = Scheduler(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=training_loader,
        validation_loader=validation_loader
    )

    # Initialize results
    best_result = BestResult()

    # Log training start
    write_log_message(f"Starting training for {args.epochs} epochs.")

    # Train and evaluate
    early_stopping_reason = ""
    for epoch in tqdm(range(1, args.epochs+1), initial=0):
        if early_stopping_reason:
            break

        # Train one epoch
        train_loss = scheduler.train_one_epoch(lr_scheduler=lr_scheduler)

        # Evaluate
        val_loss = scheduler.evaluate()

        # Add losses to graph
        writer.add_scalars('loss/overview', {'train': train_loss, 'val': val_loss}, epoch)

        # Create model folder
        weights_folder = os.path.join(log_dir, "weights/")
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

        # Update best result
        if val_loss <= best_result.loss:
            best_result.loss = val_loss
            best_result.epoch = epoch

            # Save best model
            torch.save(model.state_dict(), os.path.join(weights_folder, f"best.pt"))

        # Early stopping
        epoch_difference = epoch - best_result.epoch
        if epoch_difference > args.early_stopping >= 0:
            early_stopping_reason = "Model has not improved for {} epoch{}.".format(
                epoch_difference, 's' if epoch_difference != 1 else ''
            )

        # Save latest model
        torch.save(model.state_dict(), os.path.join(weights_folder, f"latest.pt"))

    writer.close()
    write_log_message(f"Training has been finished.", early_stopping_reason)


if __name__ == "__main__":
    train()
