# TripletLoss

## General

This project is a deep learning implementation for training a triplet loss-based model using PyTorch. The model is designed to learn embeddings for images, which can be used for several tasks.

## Features
- **Data Augmentation**: Optional data augmentation can be applied during training to improve model robustness.
- **TensorBoard Logging**: Training and validation metrics are logged using TensorBoard for visualization.
- **Hyperparameter Saving**: Hyperparameters are saved in a YAML file for reproducibility.
- **Evaluation**: Evaluation script to test the performance of the trained model.
- **Model Support**: The framework supports multiple models (ResNeXt50 and ViT_B).

## Requirements
- Python 3.9.0+
- torch
- torchvision
- tensorboard
- tqdm
- pypdfium2
- numpy
- pillow
- pyyaml

## Installation
To install the required dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training the model
To train the model, run the `train.py` script with the necessary arguments. Here's an example:

```bash
python train.py -t path/to/train/dataset -v path/to/val/dataset -e 50 -l 0.0001 -b 8 -m 1.0
```

### Evaluating the model
To evaluate the model, run the `val.py` script with the necessary arguments. Here's an example:

```bash
python val.py -p path/to/test/dataset -m path/to/model
```

### Running inference with the model
To run the inference script, use the following command:

```bash
python predict.py -i path/to/input/directory -o path/to/output/directory -m path/to/model -t 0.20
```

## Overview

### `train.py`
**Main Script**: Handles the training of the image embedding model.

**Arguments**:
- `-e`, `--epochs`: Number of epochs to train for (default: 50).
- `-t`, `--train-set`: Path to the training dataset directory.
- `-v`, `--val-set`: Path to the validation dataset directory.
- `-b`, `--batch-size`: Batch size (default: 16).
- `-s`, `--image-size`: Size of the input images (default: 224).
- `-n`, `--num-workers`: Number of workers for the dataloader (default: 2).
- `-m`, `--margin`: Margin value for the triplet loss function (default: 1.5).
- `-l`, `--learning-rate`: Initial learning rate for the optimizer (default: 0.0001).
- `--target-lr`: Learning rate at the end of the training. If not specified, the learning rate is fixed (default: None).
- `--lr-steps`: The epoch in which the learning rate should be adjusted. The new learning rate is calculated by the difference between the initial learning rate and the target learning rate divided by the number of steps (default: None).
- `--lr-schedule`: Type of learning rate schedule. Choose from 'fixed', 'linear', 'exponential' or 'steps' (default: 'fixed').
- `--positive-mining-strategy`: The mining strategy for positive samples. Choose from 'random' or 'easy' (default: 'random').
- `--negative-mining-strategy`: The mining strategy for negative samples. Choose from 'random', 'semi-hard' or 'hard' (default: 'semi-hard').
- `--loss-function`: Loss function to use. Choose from 'TripletLoss' or 'SCTLoss' (default: 'TripletLoss').
- `--temperature`: Temperature for the positive and negative scores. Only relevant when using SCTLoss (default: 0.1).
- `--seed`: Random seed for reproducibility (default: 42).
- `--augment`: Whether to apply data augmentation during the training (default: False).
- `--early-stopping`: Number of epochs without improvement before stopping the training. Set to -1 to disable (default: -1).
- `--pretrained-weights`: Path to the pre-trained weights (default: None).
- `--log-folder`: Directory where logs and other outputs will be saved (default: './runs').
- `--deterministic-algorithms`: Whether deterministic algorithms should be used during training (default: False).
- `--model-name`: Model architecture to train. Choose from 'ResNeXt50' or 'ViT_B' (default: 'ResNeXt50').

### `val.py`
**Evaluation Script**: Handles the evaluation of the trained model.

**Arguments**:
- `-p`, `--path`: Path to the test dataset directory.
- `-m`, `--model`: Path to the trained model.
- `-s`, `--image-size`: Size of the input images (default: 224).
- `-b`, `--batch-size`: Batch size (default: 16).
- `-n`, `--num-workers`: Number of workers for data loading (default: 4).
- `-t`, `--threshold`: The threshold for calculating the metrics. If not provided, the best threshold will be calculated (default: None).
- `--seed`: Seed for reproducibility (default: 42).
- `--model-name`: Model architecture to use. Choose from 'ResNeXt50' or 'ViT_B' (default: `ResNeXt50`).

### `predict.py`
**Inference Script**: Handles the prediction and clustering of files using the trained model.

**Arguments**:
- `-i`, `--input-path`: Path to the input directory containing files to be sorted.
- `-o`, `--output-path`: Path to the output directory where sorted files will be stored.
- `-m`, `--model`: Path to the trained model.
- `-t`, `--threshold`: Threshold for clustering files. The best threshold can be calculated by `val.py`.
- `-s`, `--image-size`: Size of the input images (default: 224).
- `-b`, `--batch-size`: Batch size (default: 16).
- `-n`, `--num-workers`: Number of workers for data loading (default: 4).
- `--seed`: Seed for reproducibility (default: 42).
- `--model-name`: Model architecture to use. Choose from 'ResNeXt50' or 'ViT_B' (default: `ResNeXt50`).
- `--clustering-strategy`: Choose the clustering strategy from "all" or "avg". "all" means that every distance between a file and the files in an existing cluster must be lower than or equal to the threshold. "avg" means that the average distance between a file and the files in an existing cluster must be lower than or equal to the threshold (default: `avg`).

### Example Usage


## Data
The data should be sorted as shown below. If possible, each template should also contain at least two examples, as a positive sample is assigned to each anchor.
```
.
├── train/
│   │ 
│   ├── template_1/
│   │   │ 
│   │   ├── file_1.ext
│   │   ├── file_2.ext
│   │   └── ...
│   │ 
│   ├── template_2/
│   │   │ 
│   │   ├── file_1.ext
│   │   ├── file_2.ext
│   │   └── ...
│   │ 
│   └── ...
│       │ 
│       ├── file_1.ext
│       ├── file_2.ext
│       └── ...
│ 
├── val/
│   │ 
│   ├── template_1/
│   │   │ 
│   │   ├── file_1.ext
│   │   ├── file_2.ext
│   │   └── ...
│   │ 
│   ├── template_2/
│   │   │ 
│   │   ├── file_1.ext
│   │   ├── file_2.ext
│   │   └── ...
│   │ 
│   └── ...
│       │ 
│       ├── file_1.ext
│       ├── file_2.ext
│       └── ...
│ 
└── test/
    │ 
    ├── template_1/
    │   │ 
    │   ├── file_1.ext
    │   ├── file_2.ext
    │   └── ...
    │ 
    ├── template_2/
    │   │ 
    │   ├── file_1.ext
    │   ├── file_2.ext
    │   └── ...
    │ 
    └── ...
        │ 
        ├── file_1.ext
        ├── file_2.ext
        └── ...
```
