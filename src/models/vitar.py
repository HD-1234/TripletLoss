from collections import OrderedDict
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseEmbeddingModel

import torch


__all__ = ["VisionTransformerAnyResolutionB16", "VisionTransformerAnyResolutionL16"]


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
    def __init__(self, hidden_dim: int, num_heads: int, num_tokens: int) -> None:
        """
        Initializes the Grid Attention layer.

        Args:
            hidden_dim (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            num_tokens (int): The number of tokens per image dimension.
        """
        super(GridAttention, self).__init__()
        self.num_tokens = num_tokens

        # Initialize the cross attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def _preprocess(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Preprocess the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, width, height, d_model).

        Returns:
            Tuple: The query and the key/value of shape (batch_size, seq_length, d_model).
        """
        # Get batch size, width, height and d_model
        batch_size, d_model, width, height = x.shape

        assert width == height

        # Calculate the size of each grid cell
        grid_size = height // self.num_tokens

        # Apply average pooling to each grid
        q = nn.AdaptiveAvgPool2d((self.num_tokens, self.num_tokens))(x)

        # (batch_size, d_model, width, height) -> (batch_size, d_model, seq_length)
        q = q.flatten(2)

        # (batch_size, d_model, seq_length) -> (batch_size, seq_length, d_model)
        q = q.permute(0, 2, 1)

        # (batch_size, d_model, width, height) -> (batch_size, width, height, d_model)
        k_v = x.permute(0, 2, 3, 1)

        # Build the grid
        k_v = k_v.unfold(1, grid_size, grid_size).unfold(2, grid_size, grid_size)
        k_v = k_v.contiguous().view(x.size())

        # (batch_size, height, width, d_model) -> (batch_size, seq_length, d_model)
        k_v = k_v.reshape(batch_size, -1, d_model)

        return q, k_v

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Grid Attention Module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, d_model, width, height).

        Returns:
            Tensor: The output tensor with shape (batch_size, seq_length, d_model)
        """
        # Get query, key and value
        q, k_v = self._preprocess(x)

        # Apply cross attention layer
        output = torch.cat(
            [
                self.cross_attention(q[:, ind, :].unsqueeze(1), c, c)[0]
                for ind, c in enumerate(k_v.chunk(self.num_tokens ** 2, dim=1))
            ],
            dim=1
        )

        return output


class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the Adaptive Token Merger.

        Args:
            num_tokens (int): The number of tokens per image dimension.
            num_heads (int): The number of attention heads.
            d_model (int): The hidden dimension.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Grid attention
        self.grid_attention = GridAttention(hidden_dim=d_model, num_heads=num_heads, num_tokens=num_tokens)

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
        # Apply Grid Attention
        x = self.grid_attention(x)

        # Pass through the feed-forward network
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
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

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
        x = self.self_attention(x, x, x)[0]

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
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 num_tokens: int = 14,
                 d_model: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 ff_dim: int = 2048,
                 mlp_dim: int = 3072,
                 dropout: float = 0.0,
                 ) -> None:
        """
        Initializes the Vision Transformer Any Resolution (ViTAR).

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patches.
            num_tokens (int): The number of tokens per image dimension.
            d_model (int): The hidden dimension.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of encoder layers.
            ff_dim (int): The dimension of the feed-forward network.
            mlp_dim (int): The dimension of the mlp network.
            dropout (float): The dropout probability.
        """
        super(VisionTransformerAnyResolution, self).__init__()
        self.num_tokens = num_tokens

        # Convolutional layer to create the patches
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )

        # Calculate the sequence length including the class token
        seq_length = num_tokens ** 2

        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.zeros(seq_length, d_model))

        # Initialize the positional encoding parameters using Xavier uniform initialization
        nn.init.xavier_uniform_(self.pos_encoding)

        # Create class tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Calculate the sequence length including the class token
        seq_length += 1

        # Adaptive Token Merger (ATM)
        self.adaptive_token_merger = AdaptiveTokenMerger(
            num_tokens=num_tokens, num_heads=num_heads, d_model=d_model, ff_dim=ff_dim, dropout=dropout
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
        size = x.size(-1)

        # Get the scale factor
        scale_factor = size // self.num_tokens

        # (seq_length, d_model) -> (width, height, d_model)
        pos = self.pos_encoding.reshape(self.num_tokens, self.num_tokens, -1)

        # (width, height, d_model) -> (d_model, width, height)
        pos = pos.permute(2, 0, 1)

        if fuzzy_positional_encoding:
            # Generate a tensor filled with random numbers from a uniform distribution between -0.5 and 0.5
            rnd_num = torch.rand(pos.size(), dtype=pos.dtype, device=pos.device) - 0.5

            # Add fuzziness to the positional encodings
            pos = pos.add(rnd_num)

        # (d_model, width, height) -> (1, d_model, width, height)
        pos = pos.unsqueeze(0)

        # (1, d_model, width, height) -> (batch_size, d_model, width, height)
        pos = pos.repeat(x.size(0), 1, scale_factor, scale_factor)

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

        # Apply the Adaptive token merger with Grid Attention
        x = self.adaptive_token_merger(x)

        # Get the batch size from the input tensor
        batch_size = x.shape[0]

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

        # Apply global average pooling
        out = out.mean(dim=1)

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
