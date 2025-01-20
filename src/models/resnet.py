from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseEmbeddingModel


__all__ = ["ResNet50", "ResNet101", "ResNet152", "ResNeXt50", "ResNeXt101"]


class ResNeXt(nn.Module):
    def __init__(self, groups: int = 32, width_per_group: int = 4, blocks: Optional[List] = None) -> None:
        """
        Initializes the ResNeXt architecture.

        Args:
            groups (int): Number of groups in each block.
            width_per_group (int): Width per group.
            blocks (Optional[List]): The number of blocks in each layer.
        """
        super(ResNeXt, self).__init__()

        # Define the initial number of input channels and the expansion
        self.in_channels = 64
        self.expansion = 4

        # Store group and width per group
        self.groups = groups
        self.width_per_group = width_per_group

        # Convolutional layer with kernel size 7x7, stride 2, padding 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Batch normalization layer
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # Max-pooling layer with kernel size 3x3, stride 2 and padding 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Set the number of blocks in each layer if not provided
        if not blocks:
            blocks = [3, 4, 6, 3]

        # Define the ResNeXt blocks
        self.layer1 = self.build_layer(out_channels=64, num_blocks=blocks[0], stride=1)
        self.layer2 = self.build_layer(out_channels=128, num_blocks=blocks[1], stride=2)
        self.layer3 = self.build_layer(out_channels=256, num_blocks=blocks[2], stride=2)
        self.layer4 = self.build_layer(out_channels=512, num_blocks=blocks[3], stride=2)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layer with the output size 1000
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

    def build_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Builds a ResNeXt block.

        Args:
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride value.

        Returns:
            nn.Sequential: The ResNeXt block.
        """
        layers = list()

        # Add the first block with down sampling
        layers.append(
            Bottleneck(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
                expansion=self.expansion,
                downsampling=True
            )
        )

        # Update the number of input channels for the next number of blocks
        self.in_channels = out_channels * self.expansion

        # Add the blocks without down sampling
        for i in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    expansion=self.expansion
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ResNeXt model.

        Args:
            x (Tensor): The inout tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Convolutional layer
        x = self.conv1(x)

        # Batch normalization and ReLU activation
        x = self.bn1(x)
        x = self.relu(x)

        # Max pooling layer
        x = self.maxpool(x)

        # ResNeXt blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive average pooling layer
        x = self.avgpool(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            groups: int = 32,
            width_per_group: int = 4,
            expansion: int = 4,
            downsampling: bool = False
    ) -> None:
        """
        Initializes a ResNeXt bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value.
            groups (int): Number of groups.
            width_per_group (int): Width per group.
            downsampling (bool): Whether to down sample.
        """
        super().__init__()

        # Calculate the width of each group
        width = int(out_channels * (width_per_group / 64.0)) * groups

        # Store whether to down sample or not
        self.downsampling = downsampling

        # Convolutional layer with kernel size 1x1
        self.conv1 = self.conv1x1(in_channels=in_channels, out_channels=width)

        # Batch normalization layer
        self.bn1 = self.norm_layer(width)

        # Convolutional layer with kernel size 3x3
        self.conv2 = self.conv3x3(in_channels=width, out_channels=width, stride=stride, groups=groups)

        # Batch normalization layer
        self.bn2 = self.norm_layer(width)

        # Convolutional layer with kernel size 1x1
        self.conv3 = self.conv1x1(in_channels=width, out_channels=out_channels * expansion)

        # Batch normalization layer
        self.bn3 = self.norm_layer(out_channels * expansion)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # Down sampling layer if required
        if self.downsampling:
            self.downsample = nn.Sequential(
                self.conv1x1(in_channels, out_channels * expansion, stride=stride),  # 1x1 Convolutional layer
                self.norm_layer(out_channels * expansion)  # Batch normalization layer
            )

    @staticmethod
    def conv3x3(in_channels: int, out_channels: int, stride: int, groups: int, padding: int = 1,
                dilation: int = 1) -> nn.Conv2d:
        """
        Creates a 3x3 convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value.
            groups (int): Number of groups.
            padding (int): Padding value.
            dilation (int): Dilation value.

        Returns:
            nn.Conv2d: The created convolutional layer.
        """
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

    @staticmethod
    def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
        """
        Creates a 1x1 convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value.

        Returns:
            nn.Conv2d: The created convolutional layer.
        """
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=False
        )

    @staticmethod
    def norm_layer(num_features: int) -> nn.BatchNorm2d:
        """
        Creates a batch normalization layer.

        Args:
            num_features (int): Number of features.

        Returns:
            nn.BatchNorm2d: The created batch normalization layer.
        """
        return nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Store input tensor
        original = x

        # Convolutional layer
        out = self.conv1(x)

        # Batch normalization and ReLU activation
        out = self.bn1(out)
        out = self.relu(out)

        # Convolutional layer
        out = self.conv2(out)

        # Batch normalization and ReLU activation
        out = self.bn2(out)
        out = self.relu(out)

        # Convolutional layer
        out = self.conv3(out)

        # Batch normalization
        out = self.bn3(out)

        # Down sampling if required
        if self.downsampling:
            original = self.downsample(x)

        # Concatenate input and output tensor
        out += original

        # ReLU activation
        out = self.relu(out)

        return out


class ResNeXt101(BaseEmbeddingModel):
    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        """
        Initializes the ResNeXt101 64x4d model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
        """
        super(ResNeXt101, self).__init__(pretrained_weights=pretrained_weights)

    def _initialize_model(self) -> nn.Module:
        """
        Initializes the specific embedding model

        Returns:
            nn.Module: The initialized model.
        """
        return ResNeXt(groups=64, width_per_group=4, blocks=[3, 4, 23, 3])

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = torch.nn.Linear(num_features, 128)


class ResNeXt50(BaseEmbeddingModel):
    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        """
        Initializes the ResNeXt50 32x4d model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
        """
        super(ResNeXt50, self).__init__(pretrained_weights=pretrained_weights)

    def _initialize_model(self) -> nn.Module:
        """
        Initializes the specific embedding model

        Returns:
            nn.Module: The initialized model.
        """
        return ResNeXt(groups=32, width_per_group=4, blocks=[3, 4, 6, 3])

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = torch.nn.Linear(num_features, 128)


class ResNet152(BaseEmbeddingModel):
    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        """
        Initializes the ResNet152 model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
        """
        super(ResNet152, self).__init__(pretrained_weights=pretrained_weights)

    def _initialize_model(self) -> nn.Module:
        """
        Initializes the specific embedding model

        Returns:
            nn.Module: The initialized model.
        """
        return ResNeXt(groups=1, width_per_group=64, blocks=[3, 8, 36, 3])

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = torch.nn.Linear(num_features, 128)


class ResNet101(BaseEmbeddingModel):
    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        """
        Initializes the ResNet101 model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
        """
        super(ResNet101, self).__init__(pretrained_weights=pretrained_weights)

    def _initialize_model(self) -> nn.Module:
        """
        Initializes the specific embedding model

        Returns:
            nn.Module: The initialized model.
        """
        return ResNeXt(groups=1, width_per_group=64, blocks=[3, 4, 23, 3])

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = torch.nn.Linear(num_features, 128)


class ResNet50(BaseEmbeddingModel):
    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        """
        Initializes the ResNet50 model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
        """
        super(ResNet50, self).__init__(pretrained_weights=pretrained_weights)

    def _initialize_model(self) -> nn.Module:
        """
        Initializes the specific embedding model

        Returns:
            nn.Module: The initialized model.
        """
        return ResNeXt(groups=1, width_per_group=64, blocks=[3, 4, 6, 3])

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = torch.nn.Linear(num_features, 128)
