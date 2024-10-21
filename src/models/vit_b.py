from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Initializes the Multi-Head Attention layer.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if self.head_dim * num_heads != d_model:
            raise ValueError("d_model must be divisible by num_heads")

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Linear projection for the output
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Splits the last dimension into (num_heads, head_dim) and transposes the result.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor with shape (batch_size, num_heads, seq_length, head_dim).
        """
        # Get batch_size and seq_length
        batch_size, seq_length = x.size(0), x.size(1)

        # Reshape the tensor to (batch_size, num_heads, head_dim)
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to shape (batch_size, num_heads, seq_length, head_dim)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: Tensor) -> Tensor:
        """
        Combines the heads and makes the tensor contiguous.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor with shape (batch_size, seq_length, d_model).
        """
        # Get batch_size and seq_length
        batch_size, seq_length = x.size(0), x.size(2)

        # Transpose to shape (batch_size, seq_length, num_heads, head_dim)
        x = x.transpose(1, 2)

        # Make the tensor contiguous
        x = x.contiguous()

        # Reshape the tensor to (batch_size, seq_length, d_model)
        x = x.view(batch_size, seq_length, self.d_model)

        return x

    def score_function(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Calculates the attention scores.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.

        Returns:
            Tensor: The attention output.
        """
        # Calculate the attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply the softmax function
        attn = torch.softmax(attn, dim=-1)

        # Apply the attention weights to the value tensor
        out = torch.matmul(attn, v)

        return out

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Multi-Head Attention layer.

        Args:
            x (Tensor): The input tensor with shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: The output tensor with shape (batch_size, seq_length, d_model).
        """
        # Project the input to get query, key, and value tensors
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the heads and transpose the result
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Calculate the attention scores
        attn_output = self.score_function(q, k, v)

        # Combine the heads and make the tensor contiguous
        attn_output = self.combine_heads(attn_output)

        # Project the output back to the original dimension
        output = self.out_proj(attn_output)

        return output


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
            ('1', nn.GELU(approximate='none')),
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
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 mlp_dim: int,
                 dropout: float = 0.0
                 ) -> None:
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
        self.ln_1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6, elementwise_affine=True)

        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.ln_2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6, elementwise_affine=True)

        # MLP block
        self.mlp = MLPBlock(d_model, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EncoderBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Store input tensor
        original = x

        # Apply layer normalization
        x = self.ln_1(x)

        # Calculate self-attention
        x = self.self_attention(x)

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
    def __init__(self,
                 seq_length: int,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 mlp_dim: int,
                 dropout: float = 0.0
                 ) -> None:
        """
        Initializes the Encoder.

        Args:
            seq_length (int): The sequence length.
            num_layers (int): The number of encoder layers.
            num_heads (int): The number of attention heads.
            d_model (int): The dimension of the model.
            mlp_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(Encoder, self).__init__()

        # Positional embedding for the input sequence
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, d_model).normal_(std=0.02))  # from BERT

        # Dropout layer to apply after adding positional embeddings
        self.dropout = nn.Dropout(dropout)

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
        self.ln = nn.LayerNorm(normalized_shape=d_model, eps=1e-6, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Add positional embeddings to the input tensor
        x = x + self.pos_embedding

        # Apply dropout to the input tensor with positional embeddings
        x = self.dropout(x)

        # Pass the input through the encoder layers
        x = self.layers(x)

        # Apply layer normalization to the output of the encoder layers
        x = self.ln(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_dim: int = 3072,
                 dropout: float = 0.0,
                 ) -> None:
        """
        Initializes the Vision Transformer.

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patches.
            embedding_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of encoder layers.
            mlp_dim (int): The dimension of the mlp network.
            dropout (float): The dropout probability.
        """
        super(VisionTransformer, self).__init__()

        # Convolutional layer to create the patches
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        # Create class tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Calculate the sequence length including the class token
        seq_length = (img_size // patch_size) ** 2
        seq_length += 1

        # Initialize the encoder
        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=embedding_dim,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Fully connected layer for the final output
        self.heads = nn.Sequential(OrderedDict([
            ('head', nn.Linear(embedding_dim, 1000))
        ]))

    def create_embeddings(self, image: Tensor) -> Tensor:
        """
        Creates the image embeddings.

        Args:
            image (Tensor): The input image.

        Returns:
            Tensor: The image embeddings.
        """
        # Image to patches
        emb = self.conv_proj(image)

        # Flatten the patches to a 2D tensor
        emb = emb.flatten(2)

        # Transpose to Batch size, sequence length and embedding dimension
        emb = emb.transpose(1, 2)

        return emb

    def preprocess(self, x: Tensor) -> Tensor:
        """
        Preprocesses the input tensor by creating embeddings and adding the class token.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The preprocessed tensor.
        """
        # Create embeddings from the input images
        emb = self.create_embeddings(x)

        # Get the batch size from the input tensor
        batch_size = x.shape[0]

        # Expand the class token to match the embedding tensors
        class_tokens = self.class_token.expand(batch_size, -1, -1)

        # Concatenate the class token with the embeddings
        out = torch.cat([class_tokens, emb], dim=1)

        return out

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Create embeddings and add a class token
        out = self.preprocess(x)

        # Run encoder
        out = self.encoder(out)

        # Extract the class token
        out = out[:, 0]

        # Returns the image embedding
        out = self.heads(out)

        return out


class VisionTransformerB(nn.Module):
    def __init__(self, pretrained_weights: str = None, image_size: int = 224) -> None:
        """
        Initializes the Vision Transformer model.

        Args:
            pretrained_weights (str): Path to the pre-trained weights.
            image_size (int): The size of the input image.
        """
        super(VisionTransformerB, self).__init__()

        # Initialize the Vision Transformer model
        self.embedding_model = VisionTransformer(img_size=image_size)

        # Load pre-trained weights if provided
        if pretrained_weights is not None:
            self.embedding_model.load_state_dict(torch.load(pretrained_weights, weights_only=True))

        # Replace the last layer with a new one
        num_features = self.embedding_model.heads.head.in_features
        self.embedding_model.heads.head = nn.Linear(num_features, 128)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the document embedding model.

        Args:
            anchor (Tensor): Anchor tensor.
            positive (Tensor): Positive tensor.
            negative (Tensor): Negative tensor.

        Returns:
            Tuple: The output tensors.
        """
        # Pass the anchor, positive, and negative tensors through the Vision Transformer model
        anchor = self.embedding_model(anchor)
        positive = self.embedding_model(positive)
        negative = self.embedding_model(negative)

        return anchor, positive, negative
