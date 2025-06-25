from typing import Tuple

import torch.nn as nn
from torch import Tensor


class BaseEmbeddingModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Initializes the base embedding model.

        Args:
            **kwargs: Additional arguments specific to the model.
        """
        super(BaseEmbeddingModel, self).__init__()

        # Initialize the specific model
        self.embedding_model = self._initialize_model(**kwargs)

        # Replace the last layer with a new one
        self._replace_last_layer()

    def _initialize_model(self, **kwargs) -> nn.Module:
        """
        Initializes the specific embedding model

        Args:
            **kwargs: Additional arguments specific to the model.

        Returns:
            nn.Module: The initialized model.
        """
        raise NotImplementedError("Subclasses must implement a '_initialize_model' method.")

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        raise NotImplementedError("Subclasses must implement a '_replace_last_layer' method.")

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
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
        anchor = self.embedding_model(anchor, **kwargs)
        positive = self.embedding_model(positive, **kwargs)
        negative = self.embedding_model(negative, **kwargs)

        return anchor, positive, negative
