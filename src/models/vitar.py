from collections import OrderedDict

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit import MultiHeadAttention
from src.models.base_model import BaseEmbeddingModel

import torch


__all__ = ["VisionTransformerAnyResolutionB16", "VisionTransformerAnyResolutionL16"]


class GridPadding(nn.Module):
    def __init__(self, grid_size: int, num_tokens: int) -> None:
        """
        Initializes the Grid Padding.

        Args:
            num_tokens (int): The final number of tokens per dimension.
            grid_size (int): The size of each grid cell.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = num_tokens

        # Validate initialization parameters
        if grid_size <= 0 or num_tokens <= 0:
            raise ValueError("grid_size and num_tokens must be positive integers")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the grid padding to the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, d_model, width, height).

        Returns:
            Tensor: The padded tensor of shape (batch_size, d_model, grid_size * num_tokens, grid_size * num_tokens).
        """
        # Get current dimensions and validate square input
        batch_size, d_model, width, height = x.shape
        assert width == height

        # Skip padding if the input tensor already matches target dimensions
        if width % self.num_tokens == 0:
            return x

        # Calculate the target dimensions
        target_dim = self.grid_size * self.num_tokens

        # Up-sampling using 2D transposed convolution
        weights = torch.ones((d_model, 1, 1, 1), dtype=x.dtype, device=x.device)
        output = F.conv_transpose2d(x, weight=weights, stride=self.grid_size, groups=d_model)

        # Get the up-sampled dimension
        dim = output.shape[-1]

        # Get the original indices
        indices = torch.arange(start=0, end=dim, step=self.grid_size, device=x.device)

        # Create a mask to keep only the required padding cells to match the target dimension
        mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
        mask[indices] = True

        # Get the new indices
        padding_indices = torch.arange(start=0, end=dim, step=1, device=x.device)[~mask]

        # Calculate the max index and update the mask
        padding_dim = target_dim - width
        pad_indices_keep = padding_indices[:padding_dim]
        mask[pad_indices_keep] = True

        # Create 2D mask
        mask = mask.unsqueeze(0) & mask.unsqueeze(-1)

        # Apply mask to output tensor
        output = output[:, :, mask]

        # (batch_size, d_model, width * height) -> (batch_size, d_model, width, height)
        output = output.reshape(batch_size, d_model, target_dim, target_dim)

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        """
        Initializes position-wise fully connected feed-forward network.

        Args:
            d_model (int): The dimension of the model.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GridAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, grid_size: int) -> None:
        """
        Initializes the Grid Attention layer.

        Args:
            hidden_dim (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            grid_size (int): The size of each grid cell.
        """
        super(GridAttention, self).__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Initialize the cross attention layer
        self.cross_attention = MultiHeadAttention(d_model=hidden_dim, num_heads=num_heads)

        # Average pooling layer
        self.avg_pooling_layer = nn.AvgPool2d((self.grid_size, self.grid_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Grid Attention Module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, d_model, width, height).

        Returns:
            Tensor: The output tensor with the shape (batch_size, seq_length, d_model)
        """
        # Get batch size, width, height and d_model
        batch_size, d_model, width, height = x.shape
        assert width == height

        # Get the number of grid cells for each dimension
        g_h, g_w = height // self.grid_size, width // self.grid_size
        num_grid_cells = g_h * g_w

        # Apply average pooling to each grid
        q = self.avg_pooling_layer(x)

        # (batch_size, d_model, width, height) -> (batch_size, seq_length, d_model)
        q = q.flatten(2).permute(0, 2, 1)

        # (batch_size, seq_length, d_model) -> (batch_size * num_grid_cells 1, d_model)
        q = q.reshape(batch_size * num_grid_cells, d_model).unsqueeze(dim=1)

        # (batch_size, d_model, width, height) -> (batch_size, width, height, d_model)
        k_v = x.permute(0, 2, 3, 1)

        # Build the grid of shape (batch_size * num_grid_cells, GRID_SIZE * GRID_SIZE, d_model)
        k_v = k_v.unfold(1, self.grid_size, self.grid_size).unfold(2, self.grid_size, self.grid_size)
        k_v = k_v.reshape(batch_size * num_grid_cells, d_model, self.grid_size * self.grid_size).permute(0, 2, 1)

        # Compute cross attention
        attention = self.cross_attention(q, k_v, k_v)

        # Residual connection
        output = q + attention

        # (batch_size * num_grid_cells, 1, d_model) -> (batch_size, num_grid_cells, d_model)
        output = output.reshape(batch_size, num_grid_cells, d_model)

        return output


class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the Adaptive Token Merger.

        Args:
            grid_size (int): The size of each grid cell.
            num_heads (int): The number of attention heads.
            d_model (int): The hidden dimension.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Grid attention
        self.grid_attention = GridAttention(hidden_dim=d_model, num_heads=num_heads, grid_size=grid_size)

        # The feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model=d_model, ff_dim=ff_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Adaptive Token Merger.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Apply grid attention
        x = self.grid_attention(x)

        # Pass attention output through the feed-forward network
        x = self.feed_forward(x)

        return x


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int, dropout: float) -> None:
        """
        Initializes the MLPBlock.

        Args:
            d_model (int): The dimension of the model.
            mlp_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(MLPBlock, self).__init__()

        # Define the layers of the MLP block using an OrderedDict
        self._modules = OrderedDict([
            ('0', nn.Linear(d_model, mlp_dim, bias=True)),
            ('1', nn.GELU()),
            ('2', nn.Dropout(dropout)),
            ('3', nn.Linear(mlp_dim, d_model, bias=True)),
            ('4', nn.Dropout(dropout))
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLPBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Iterate through the layers and apply them sequentially
        for i in range(5):
            x = self._modules[str(i)](x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, d_model: int, mlp_dim: int, dropout: float = 0.0) -> None:
        """
        Initializes the EncoderBlock.

        Args:
            num_heads (int): The number of attention heads.
            d_model (int): The dimension of the model.
            mlp_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(EncoderBlock, self).__init__()

        # Layer normalization
        self.ln_1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.ln_2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        # MLP block
        self.mlp = MLPBlock(d_model, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: The output tensor.
        """
        # Store input tensor
        original = x

        # Apply layer normalization
        x = self.ln_1(x)

        # Calculate self-attention
        x = self.self_attention(x, x, x)

        # Apply dropout
        x = self.dropout(x)

        # Concatenate input and attention output
        x = x + original

        # Apply layer normalization
        y = self.ln_2(x)

        # Pass through the MLP block
        y = self.mlp(y)

        return x + y


class Encoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            d_model: int,
            mlp_dim: int,
            dropout: float = 0.0
    ) -> None:
        """
        Initializes the Encoder.

        Args:
            num_layers (int): The number of encoder layers.
            num_heads (int): The number of attention heads.
            d_model (int): The dimension of the model.
            mlp_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(Encoder, self).__init__()

        # Initialize the layers of the encoder
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f'encoder_layer_{i}'] = EncoderBlock(
                num_heads=num_heads,
                d_model=d_model,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
        self.layers = nn.Sequential(layers)

        # Layer normalization to apply after the encoder layers
        self.ln = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Pass the input through the encoder layers
        x = self.layers(x)

        # Apply layer normalization to the output of the encoder
        x = self.ln(x)

        return x


class VisionTransformerAnyResolution(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_tokens: int = 14,
        grid_size: int = 2,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 2048,
        mlp_dim: int = 3072,
        dropout: float = 0.0
    ) -> None:
        """
        Initializes the Vision Transformer Any Resolution (ViTAR).

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patches.
            num_tokens (int): The number of tokens per image dimension.
            grid_size (int): The size of each grid cell.
            d_model (int): The hidden dimension.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of encoder layers.
            ff_dim (int): The dimension of the feed-forward network.
            mlp_dim (int): The dimension of the mlp network.
            dropout (float): The dropout probability.
        """
        super(VisionTransformerAnyResolution, self).__init__()
        self.num_tokens = num_tokens
        self.grid_size = grid_size

        # Convolutional layer to create the patches
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )

        # Calculate the sequence length including the class token
        seq_length = num_tokens ** 2

        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model).normal_(std=0.02))

        # Initialize the positional encoding parameters using Xavier uniform initialization
        nn.init.xavier_uniform_(self.pos_encoding)

        # Grid Padding
        self.grid_padding = GridPadding(grid_size=self.grid_size, num_tokens=self.num_tokens)

        # Create class tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Calculate the sequence length including the class token
        seq_length += 1

        # Adaptive Token Merger (ATM)
        self.adaptive_token_merger = AdaptiveTokenMerger(
            grid_size=grid_size,
            num_heads=num_heads,
            d_model=d_model,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # Initialize the encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Fully connected layer for the final output
        self.heads = nn.Sequential(OrderedDict([
            ('head', nn.Linear(d_model, 1000))
        ]))

    def add_pos_encoding(self, x: Tensor, fuzzy_positional_encoding: bool):
        """
        Adds positional encodings to an input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, d_model, width, height).
            fuzzy_positional_encoding (bool): Whether to apply fuzzy positional encodings or not.

        Returns:
            Tensor: The input tensor with the positional encoding.
        """
        # Get the current size
        width, height = x.shape[-2:]

        assert width == height

        # (seq_length, d_model) -> (width, height, d_model)
        pos = self.pos_encoding.reshape(self.num_tokens, self.num_tokens, -1)

        # (width, height, d_model) -> (1, d_model, width, height)
        pos = pos.permute(2, 0, 1).unsqueeze(0)

        if width != self.num_tokens:
            # Apply interpolation on the positional embeddings
            pos = nn.functional.interpolate(pos, size=(width, height), mode='bicubic')

        if fuzzy_positional_encoding:
            # Create a grid of coordinates of shape (width, height, 2)
            grid = torch.stack(
                torch.meshgrid(
                    torch.arange(width, device=x.device),
                    torch.arange(height, device=x.device),
                    indexing='ij'
                ),
                dim=-1
            ).to(x.dtype)

            # Add random offsets with a uniform distribution between [-0.5, 0.5]
            grid += torch.rand_like(grid) - 0.5

            # grid_sample() expects the left-top corner at [-1, -1] and the bottom-right corner at [1, 1]
            grid /= torch.tensor([width - 1, height - 1], device=x.device)

            # Handle NaN values by setting the values to 0
            grid = grid.nan_to_num(0)

            # (r2 - r1) * torch.rand(a, b) + r1 for a uniform distribution between [r1, r2].
            # Here the range is between [-1, 1]
            grid = (2 * grid) - 1

            # Clamps all elements into the range [0, width]. MPS does not support padding mode border for grid_sample().
            grid = grid.clamp(min=-1, max=1)

            # (width, height, 2) -> (1, width, height, 2)
            grid = grid.unsqueeze(0)

            # Sample positional embeddings at the grid locations
            pos = F.grid_sample(
                pos,
                grid,
                mode="bilinear",
                padding_mode="zeros" if x.device.type == "mps" else "border",
                align_corners=False
            )

        # (1, d_model, width, height) -> (batch_size, d_model, width, height)
        pos = pos.repeat(x.size(0), 1, 1, 1)

        return x + pos

    def _preprocess(self, x: Tensor, fuzzy_positional_encoding: bool) -> Tensor:
        """
        Preprocesses the input tensor.

        Args:
            x (Tensor): The input tensor.
            fuzzy_positional_encoding (bool): Whether to apply fuzzy positional encodings or not.

        Returns:
            Tensor: The preprocessed tensor.
        """
        # Image to patches
        x = self.conv_proj(x)

        # Add positional encodings
        x = self.add_pos_encoding(x, fuzzy_positional_encoding=fuzzy_positional_encoding)

        # Get batch size, width, height and d_model
        batch_size, d_model, width, height = x.shape

        while width > self.num_tokens or height > self.num_tokens:
            # Apply the Adaptive token merger with Grid Attention
            x = self.adaptive_token_merger(x)

            # (batch_size, seq_length, d_model) -> (batch_size, d_model, seq_length)
            x = x.permute(0, 2, 1)

            # Get width, height
            width = height = int(x.shape[-1] ** 0.5)

            # (batch_size, d_model, seq_length) -> (batch_size, d_model, width, height)
            x = x.reshape(batch_size, d_model, width, height)

            # Apply grid padding
            if 2 * self.num_tokens > width > self.num_tokens or 2 * self.num_tokens > height > self.num_tokens:
                x = self.grid_padding(x)

        # (batch, d_model, width, height) -> (batch, seq_length, d_model)
        x = x.flatten(2).permute(0, 2, 1)

        # Expand the class token to match the embedding tensors
        class_tokens = self.class_token.expand(batch_size, -1, -1)

        # Concatenate the class token with the embeddings
        x = torch.cat([class_tokens, x], dim=1)

        return x

    def forward(self, x: Tensor, fuzzy_positional_encoding: bool = False) -> Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (Tensor): The input tensor.
            fuzzy_positional_encoding (bool): Whether to apply fuzzy positional encodings or not.

        Returns:
            Tensor: The output tensor.
        """
        # Create embeddings and add a class token
        out = self._preprocess(x, fuzzy_positional_encoding=fuzzy_positional_encoding)

        # Run encoder
        out = self.encoder(out)

        # Extract the class token
        out = out[:, 0]

        # Returns the image embedding
        out = self.heads(out)

        return out


class VisionTransformerAnyResolutionB16(BaseEmbeddingModel):
    def __init__(self, image_size: int = 224) -> None:
        """
        Initializes a ViTAR-Base model with a patch size of 16.

        Args:
            image_size (int): The size of the input image.
        """
        super(VisionTransformerAnyResolutionB16, self).__init__(image_size=image_size)

    def _initialize_model(self, image_size: int) -> nn.Module:
        """
        Initializes the specific embedding model.

        Args:
            image_size (int): The size of the input image.

        Returns:
            nn.Module: The initialized model.
        """
        return VisionTransformerAnyResolution(
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=12,
            mlp_dim=3072
        )

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.heads.head.in_features
        self.embedding_model.heads.head = nn.Linear(num_features, 128)


class VisionTransformerAnyResolutionL16(BaseEmbeddingModel):
    def __init__(self, image_size: int = 224) -> None:
        """
        Initializes a ViTAR-Large model with a patch size of 16.

        Args:
            image_size (int): The size of the input image.
        """
        super(VisionTransformerAnyResolutionL16, self).__init__(image_size=image_size)

    def _initialize_model(self, image_size: int) -> nn.Module:
        """
        Initializes the specific embedding model

        Args:
            image_size (int): The size of the input image.

        Returns:
            nn.Module: The initialized model.
        """
        return VisionTransformerAnyResolution(
            patch_size=16,
            d_model=1024,
            num_heads=16,
            num_layers=24,
            mlp_dim=4096
        )

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        num_features = self.embedding_model.heads.head.in_features
        self.embedding_model.heads.head = nn.Linear(num_features, 128)
